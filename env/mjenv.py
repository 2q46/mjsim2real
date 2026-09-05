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
    mjw_data = mjw.make_data(mj_model, nworld=num_envs, njmax=350)
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
    jax_rgb_buff = jnp.array(255 * jnp.from_dlpack(render_buff), dtype=jnp.float32)
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


@partial(jax.jit, static_argnames=["batch_size", "cube_qpos_adr", "pos_range", "qpos_noise_scale"])
def _randomize_qpos(rng_key, base_qpos, batch_size, cube_qpos_adr, pos_range, qpos_noise_scale):
    k_cube, k_arm = jax.random.split(rng_key)

    arm_noise = jax.random.uniform(
        k_arm, base_qpos.shape, minval=-qpos_noise_scale, maxval=qpos_noise_scale
    )
    
    mask = jnp.ones(base_qpos.shape[-1], dtype=jnp.float32)
    mask = mask.at[cube_qpos_adr : cube_qpos_adr + 7].set(0.0)
    
    qpos = base_qpos + (arm_noise * mask)

    cube_xy_noise = jax.random.uniform(
        k_cube, (batch_size, 2), minval=-pos_range, maxval=pos_range
    )
    qpos = qpos.at[:, cube_qpos_adr : cube_qpos_adr + 2].add(cube_xy_noise)
    
    return qpos


def reset_batch(mj_model, mjw_model, mjw_data, rng_key, cube_id, pos_range=0.05, qpos_noise_scale=0.2):
    batch_size = mjw_data.qpos.shape[0]
    init_qpos = jnp.tile(wp.to_jax(mjw_model.qpos0), (batch_size, 1))
    init_qvel = jnp.zeros_like(wp.to_jax(mjw_data.qvel))

    cube_qpos_adr = get_free_body_qpos_adr(mj_model, cube_id)
    
    randomized_qpos = _randomize_qpos(
        rng_key, init_qpos, batch_size, cube_qpos_adr, pos_range, qpos_noise_scale
    )

    wp.copy(mjw_data.qpos, wp.from_jax(randomized_qpos))
    wp.copy(mjw_data.qvel, wp.from_jax(init_qvel))
    mjw.forward(mjw_model, mjw_data)


def sample_action(mjw_data, rng_key):
    size = mjw_data.ctrl.shape
    ctrl = jax.random.uniform(rng_key, size, minval=-1.0, maxval=1.0)
    ctrl = ctrl.at[..., -1].set(0.1)  # Matches working grasp threshold
    return ctrl

@partial(jax.jit)
def scale_action_to_actuators(ctrl: jax.Array) -> jax.Array:
    
    ctrl_min = jnp.array([-1.91986, -1.74533, -1.69000, -1.65806, -2.74385, -0.17453])
    ctrl_max = jnp.array([ 1.91986,  1.74533,  1.69000,  1.65806,  2.84121,  1.74533])
    
    normalized_ctrl = (ctrl + 1.0) / 2.0
    
    return ctrl_min + normalized_ctrl * (ctrl_max - ctrl_min)


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
    prev_ctrl: jax.Array,
    tolerance: float = 0.02,
):
 
    ee_cube_dist = jnp.linalg.norm(
        current_ee_pos - current_cube_pos,
        axis=-1,
    )

    cube_goal_dist = jnp.linalg.norm(
        cube_goal_pos - current_cube_pos,
        axis=-1,
    )

    reach_reward = jnp.exp(-20.0 * ee_cube_dist)

    close_reward = jnp.exp(-100.0 * ee_cube_dist)

    gripper_cmd = ctrl[..., -1]

    is_close = ee_cube_dist < 0.02
    is_gripper_closed = gripper_cmd < 0.15

    grasp_reward = (
        is_close & is_gripper_closed
    ).astype(jnp.float32)

   
    close_gripper_reward = (
        jnp.exp(-80.0 * ee_cube_dist)
        * (jnp.maximum(0, 1 - jnp.abs(gripper_cmd - 0.1)/0.1))
    )

    cube_height = 0.015
    desired_lift = 0.1

    lift_height = current_cube_pos[..., 2] - cube_height

    lift_progress = jnp.clip(
        lift_height / desired_lift,
        0.0,
        1.0,
    )

    grasp_gate = (
        is_close & is_gripper_closed
    ).astype(jnp.float32)

    lift_reward = grasp_gate * lift_progress
    goal_reward = jnp.exp(-15.0 * cube_goal_dist)

    goal_close_reward = jnp.exp(-60.0 * cube_goal_dist)

 
    success = (
        (cube_goal_dist < tolerance)
        & (ee_cube_dist < 0.025)
        & (gripper_cmd < 0.15)
    ).astype(jnp.float32)

    action_delta = jnp.linalg.norm(
        ctrl - prev_ctrl,
        axis=-1,
    )

    action_penalty = 0.01 * action_delta

    total_reward = (
        2.0 * reach_reward
        + 1.0 * close_reward
        + 2.0 * grasp_reward
        + 1.0 * close_gripper_reward
        + 4.0 * lift_reward
        + 2.0 * goal_reward
        + 3.0 * goal_close_reward
        + 20.0 * success
        - action_penalty
    )

    return (
        total_reward,
        is_close.astype(jnp.float32),
        grasp_gate,
        success,
    )



def step_batch(cube_id, gripper_id, mjw_model, mjw_data, ctrl, goal_cube_pos, prev_ctrl, n_frames=3):

    scaled_ctrl = scale_action_to_actuators(ctrl)
    ctrl_wp = wp.from_jax(scaled_ctrl)
    prev_ctrl = wp.to_jax(prev_ctrl) 
    wp.copy(mjw_data.ctrl, ctrl_wp)
    
    for _ in range(n_frames):
        mjw.step(mjw_model, mjw_data)
        
    current_ee_pos = wp.to_jax(mjw_data.site_xpos)[:, gripper_id]
    current_cube_pos = wp.to_jax(mjw_data.xpos)[:, cube_id]
    
    reward = compute_rew(goal_cube_pos, current_cube_pos, current_ee_pos, scaled_ctrl, prev_ctrl)
    return reward


if __name__ == '__main__':
    import numpy as np

    base_key = jax.random.key(12)
    k1, k2, k3 = jax.random.split(base_key, num=3)

    mj_model, mjw_model, mjw_data = init_mujoco(2)
    render_ctx, rgb_buff = init_rendering(mj_model, 2, (128, 128))

    cube_id = get_cube_id(mj_model)
    ee_id = get_gripper_id(mj_model)

    reset_batch(
        mj_model, 
        mjw_model, 
        mjw_data, 
        k2, 
        cube_id, 
        pos_range=0.05, 
        qpos_noise_scale=0.03
    )
    goal_cube_pos = find_goal_cube_pos(mj_model, mjw_data)

    frames = []
    prev_ctrl = wp.to_jax(mjw_data.ctrl)

    for i in range(100):
        k3, base_key = jax.random.split(base_key)
        obs = render_batch(mjw_model, mjw_data, render_ctx, rgb_buff)
        action = sample_action(mjw_data, k3)
        rew = step_batch(cube_id, ee_id, mjw_model, mjw_data, action, goal_cube_pos, prev_ctrl)
        prev_ctrl = action
        frames.append(np.asarray(obs[0], dtype=np.uint8))

    media.write_video(path="vid.mp4", images=frames)