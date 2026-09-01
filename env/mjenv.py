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


def render_batch(mjw_model, mjw_data, render_ctx, render_buff):
    mjw.refit_bvh(mjw_model, mjw_data, render_ctx)
    mjw.render(mjw_model, mjw_data, render_ctx)
    mjw.get_rgb(render_ctx, camera_index=0, rgb_out=render_buff)
    jax_rgb_buff = jnp.array(255*jnp.from_dlpack(render_buff), dtype=jnp.float32)
    return jax_rgb_buff


def get_cube_id(mj_model):
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "redcube")


def get_gripper_id(mj_model):
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")


def get_free_body_qpos_adr(mj_model, body_id):

    jnt_adr = mj_model.body_jntadr[body_id]
    assert jnt_adr >= 0, "body has no joint (is it welded to the world?)"
    assert mj_model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE, (
        "expected a free joint on this body for xyz+quat randomization"
    )
    return int(mj_model.jnt_qposadr[jnt_adr])


@partial(jax.jit, static_argnames=["batch_size", "cube_qpos_adr", "pos_range"])
def _randomize_cube_xy(rng_key, base_qpos, batch_size, cube_qpos_adr, pos_range):
    xy_noise = jax.random.uniform(
        rng_key, (batch_size, 2), minval=-pos_range, maxval=pos_range
    )
    return base_qpos.at[:, cube_qpos_adr:cube_qpos_adr + 2].add(xy_noise)


def reset_batch(mj_model, mjw_model, mjw_data, rng_key, cube_id, pos_range=0.05):
    batch_size = mjw_data.qpos.shape[0]
    init_qpos = jnp.tile(wp.to_jax(mjw_model.qpos0), (batch_size, 1))
    init_qvel = jnp.zeros_like(wp.to_jax(mjw_data.qvel))

    cube_qpos_adr = get_free_body_qpos_adr(mj_model, cube_id)
    init_qpos = _randomize_cube_xy(
        rng_key, init_qpos, batch_size, cube_qpos_adr, pos_range
    )

    wp.copy(mjw_data.qpos, wp.from_jax(init_qpos))
    wp.copy(mjw_data.qvel, wp.from_jax(init_qvel))
    mjw.forward(mjw_model, mjw_data)


def sample_action(mjw_data, rng_key):
    size = mjw_data.ctrl.shape
    ctrl = jax.random.uniform(rng_key, size, minval=-1.0, maxval=1.0)
    ctrl = ctrl.at[..., -1].set(0.25)
    return ctrl


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
    ctrl: jax.Array,
    tolerance: float = 0.02,
):
    dist_ee_cube = jnp.linalg.norm(current_cube_pos - current_ee_pos, axis=-1)
    dist_cube_goal = jnp.linalg.norm(cube_goal_pos - current_cube_pos, axis=-1)

    gripper_cmd = ctrl[..., -1]
    arm_ctrl = ctrl[..., :-1]
    wrist_flex = ctrl[..., -3]

    reach_rew = 1 - jnp.tanh(3 * dist_ee_cube)
    place_rew = 1 - jnp.tanh(10 * dist_cube_goal)
    reach_close = jnp.exp(-15 * dist_ee_cube)
    gripper_closed = jnp.maximum(0, 1 - (jnp.abs(gripper_cmd-0.15)/0.06))
    reach_close_and_grasp = reach_close * gripper_closed
    is_gripping = (gripper_closed > 0.5) & (dist_ee_cube < 0.013)
    gated_place = is_gripping * place_rew
    is_success = (dist_cube_goal < tolerance)
    action_penalty = 0.0005 * jnp.sum(jnp.square(arm_ctrl), axis=-1)

    total_reward = (
            reach_rew
        + 2 * reach_close
        + 2 * reach_close_and_grasp
        + 3 * gated_place
        + 5 * is_success 
    )

    is_close = (dist_ee_cube < 0.02)

    return (
        total_reward,
        is_close.astype(jnp.int8),
        is_gripping.astype(jnp.int8),
        is_success.astype(jnp.int8),
    )


def step_batch(cube_id, gripper_id, mjw_model, mjw_data, ctrl, goal_cube_pos, n_frames=3):
    wp.copy(mjw_data.ctrl, wp.from_jax(ctrl))
    for _ in range(n_frames):
        mjw.step(mjw_model, mjw_data)
    current_ee_pos = wp.to_jax(mjw_data.site_xpos)[:, gripper_id]
    current_cube_pos = wp.to_jax(mjw_data.xpos)[:, cube_id]
    reward = compute_rew(goal_cube_pos, current_cube_pos, current_ee_pos, ctrl)
    return reward


if __name__ == '__main__':
    import numpy as np

    base_key = jax.random.key(12)
    k1, k2, k3 = jax.random.split(base_key, num=3)

    mj_model, mjw_model, mjw_data = init_mujoco(2)
    render_ctx, rgb_buff = init_rendering(mj_model, 2, (128, 128))

    cube_id = get_cube_id(mj_model)
    ee_id = get_gripper_id(mj_model)
    reset_batch(mj_model, mjw_model, mjw_data, k2, cube_id, pos_range=0.05)
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