import jax
import os
import mujoco
import mediapy as media
import jax.numpy as jnp
import mujoco_warp as mjw
import warp as wp
from functools import partial

def init_mujoco(num_envs):
    scene_dir_name = "scene.xml"
    curr_dir_name = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(curr_dir_name, scene_dir_name)
    wp.init()
    print(f"Loading MJCF: {scene_path}")
    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mjw_model = mjw.put_model(mj_model)
    mjw_data = mjw.make_data(mj_model, nworld=num_envs, njmax=200)
    return mj_model, mjw_model, mjw_data

def init_rendering(mj_model, num_envs, img_size):
    render_ctx = mjw.create_render_context(
        mj_model, nworld=num_envs, cam_res=img_size, render_rgb=True, use_shadows=True
    )
    rgb_buffer = wp.zeros(
        (num_envs, img_size[1], img_size[0]), 
        dtype=wp.vec3f,
        device="cuda"
    )
    return render_ctx, rgb_buffer

def get_cube_id(mj_model):
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")

def get_gripper_id(mj_model):
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

def render_batch(mjw_model, mjw_data, render_ctx, render_buff):
    mjw.refit_bvh(mjw_model, mjw_data, render_ctx)
    mjw.render(mjw_model, mjw_data, render_ctx)
    mjw.get_rgb(render_ctx, camera_index=0, rgb_out=render_buff)
    jax_rgb_buff = jnp.array(255*jnp.from_dlpack(render_buff), dtype=jnp.float32)
    return jax_rgb_buff

@partial(jax.jit, static_argnames=["qpos_shape", "qvel_shape"])
def get_noise_arr(rng_key, qpos_shape, qvel_shape):
    qpos_key, qvel_key = jax.random.split(rng_key, num=2)
    qpos_noise = jax.random.uniform(qpos_key, qpos_shape, minval=-0.03, maxval=0.03)
    qvel_noise = jax.random.uniform(qvel_key, qvel_shape, minval=-0.03, maxval=0.03)
    return qpos_noise, qvel_noise

def reset_batch(mjw_model, mjw_data, rng_key):   
    qpos_noise, qvel_noise = get_noise_arr(rng_key, mjw_data.qpos.shape, mjw_data.qvel.shape)
    init_qpos = wp.to_jax(mjw_model.qpos0)
    wp.copy(mjw_data.qpos, wp.from_jax(qpos_noise+init_qpos))
    wp.copy(mjw_data.qvel, wp.from_jax(qvel_noise))
    mjw.forward(mjw_model, mjw_data)

def sample_action(mjw_data, rng_key):
    size = mjw_data.ctrl.shape
    return jax.random.uniform(rng_key, size, minval=-1.0, maxval=1.0)

@partial(jax.jit, static_argnames=["goal_height"])
def set_new_height(jax_cube_pos, goal_height):
    return jax_cube_pos.at[:, 2].add(goal_height)

def find_goal_cube_pos(mj_model, mjw_data, goal_height=0.1):
    cube_id = get_cube_id(mj_model)
    wp_cube_pos = mjw_data.xpos[:, cube_id].contiguous()
    jax_cube_pos = wp.to_jax(wp_cube_pos)
    return set_new_height(jax_cube_pos, goal_height)


@partial(jax.jit, static_argnames=["tolerance"])
def compute_rew(
        cube_goal_pos: jax.Array, 
        current_cube_pos: jax.Array, 
        current_ee_pos: jax.Array, 
        tolerance=0.05
    ):
    dist_ee_cube = jnp.linalg.norm(current_cube_pos - current_ee_pos, axis=-1)
    dist_cube_goal = jnp.linalg.norm(cube_goal_pos - current_cube_pos, axis=-1)
    rew_reach =  1.5 * (1.0 - jnp.tanh(3*dist_ee_cube))
    rew_move = 1.5 * (1.0 - jnp.tanh(3*dist_cube_goal))
    is_touching = (dist_ee_cube < tolerance).astype(jnp.float32)
    is_grasped = (dist_ee_cube < 0.02).astype(jnp.float32)
    is_success = (dist_cube_goal < tolerance).astype(jnp.float32)
    success = is_success.astype(jnp.int8)
    touching = is_touching.astype(jnp.int8)
    total_reward = (rew_move + rew_reach + is_grasped + 3*is_success)
    return total_reward, touching, is_grasped, success

def step_batch(cube_id, gripper_id, mjw_model, mjw_data, ctrl, goal_cube_pos, n_frames=10):

    wp.copy(mjw_data.ctrl, wp.from_jax(ctrl))
    for i in range(n_frames):
        mjw.step(mjw_model, mjw_data)
    current_ee_pos = wp.to_jax(mjw_data.site_xpos)[:, gripper_id]
    current_cube_pos = wp.to_jax(mjw_data.xpos)[:, cube_id]
    reward = compute_rew(goal_cube_pos, current_cube_pos, current_ee_pos)
    return reward

if __name__ == '__main__':

    import numpy as np
    base_key = jax.random.key(12)
    k1, k2, k3 = jax.random.split(base_key, num=3)
    mj_model, mjw_model, mjw_data = init_mujoco(2)
    render_ctx, rgb_buff = init_rendering(mj_model, 2, (128, 128))
    reset_batch(mjw_model, mjw_data, k2)
    cube_id = get_cube_id(mj_model)
    ee_id = get_gripper_id(mj_model)
    goal_cube_pos = find_goal_cube_pos(mj_model, mjw_data)
    frames = []
    for i in range(100):
        k3, base_key = jax.random.split(base_key)
        obs = render_batch(mjw_model, mjw_data, render_ctx, rgb_buff)
        action = sample_action(mjw_data, k3)
        rew = step_batch(cube_id, ee_id, mjw_model, mjw_data, action, goal_cube_pos)
        print(rew)
        frames.append(np.asarray(obs[0], dtype=np.uint8))
    media.write_video(path="vid.mp4", images=frames)
