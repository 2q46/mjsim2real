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
        mj_model, nworld=num_envs, cam_res=img_size, render_rgb=True, use_shadows=True, background_color=(0.0, 0.0, 0.0, 1.0)
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


def get_touch_sensor_adr(mj_model, name):
    """Address of a scalar touch sensor's value in sensordata."""
    sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    assert sid >= 0, f"sensor '{name}' not found in model"
    return int(mj_model.sensor_adr[sid])


def get_gripper_touch_forces(mjw_data, fixed_adr, moving_adr):
    """
    Returns (fixed_jaw_force, moving_jaw_force), each (nworld,) — summed
    contact-normal force (N) in each fingerpad's touch-sensor zone.
    """
    sensordata = wp.to_jax(mjw_data.sensordata)  # (nworld, nsensordata)
    fixed_force = sensordata[:, fixed_adr]
    moving_force = sensordata[:, moving_adr]
    return fixed_force, moving_force


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


def reset_batch(mj_model, mjw_model, mjw_data, rng_key, cube_id, pos_range=0.03, qpos_noise_scale=0.1):

    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "grasp_ready")
    batch_size = mjw_data.qpos.shape[0]
    print("nkey:", mj_model.nkey)
    assert key_id >= 0, "keyframe 'grasp_ready' not found"

    key_qpos = jnp.array(mj_model.key_qpos[key_id], dtype=jnp.float32)  # (nq,)
    key_qvel = jnp.array(mj_model.key_qvel[key_id], dtype=jnp.float32)  # (nv,)
    key_action = jnp.array(mj_model.key_ctrl[key_id], dtype=jnp.float32)

    qpos = jnp.tile(key_qpos, (batch_size, 1))
    qvel = jnp.tile(key_qvel, (batch_size, 1))
    action = jnp.tile(key_action, (batch_size, 1))

   
    cube_qpos_adr = get_free_body_qpos_adr(mj_model, cube_id)
    '''
    qpos = _randomize_qpos(
        rng_key, qpos, batch_size, cube_qpos_adr, pos_range, qpos_noise_scale
    )
    '''
    wp.copy(mjw_data.qpos, wp.from_jax(qpos))
    wp.copy(mjw_data.qvel, wp.from_jax(qvel))
    wp.copy(mjw_data.ctrl, wp.from_jax(action))
    wp.synchronize()
    mjw.forward(mjw_model, mjw_data)


def sample_action(mjw_data, rng_key):
    size = mjw_data.ctrl.shape
    ctrl = jax.random.uniform(rng_key, size, minval=-1.0, maxval=1.0)
    ctrl = ctrl.at[..., -1].set(0.1)  # Matches working grasp threshold
    return ctrl

@partial(jax.jit)
def scale_action_to_actuators(ctrl: jax.Array) -> jax.Array:

    ctrl_min = jnp.array([-1.91986, -1.74533, -1.69000, -1.65806, -2.74385, -0.17453])
    ctrl_max = jnp.array([ 1.91986,  1.74533,  1.69000,  1.65806,  2.84121,  0.65])

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

@partial(jax.jit, static_argnames=["tolerance", "touch_threshold"])
def compute_rew(
    cube_goal_pos, current_cube_pos, current_ee_pos,
    ctrl, prev_ctrl,
    fixed_jaw_force, moving_jaw_force,
    tolerance=0.02,
    touch_threshold=1.5,
):
    ee_cube_dist = jnp.linalg.norm(current_ee_pos - current_cube_pos, axis=-1)
    cube_goal_dist = jnp.linalg.norm(cube_goal_pos - current_cube_pos, axis=-1)
    raw_gripper_cmd = ctrl[..., -1]
    gripper_closed = 1.0 - jnp.tanh(5.0 * jnp.maximum(0.0, jnp.abs(raw_gripper_cmd - 0.15)/0.05))

    is_touch = (ee_cube_dist < 0.017).astype(jnp.int32)

    is_gripped = (
        (fixed_jaw_force > touch_threshold) & (moving_jaw_force > touch_threshold) & is_touch & (current_cube_pos[..., 2] > 0.02)
    ).astype(jnp.int32)
    is_success = (cube_goal_dist < tolerance).astype(jnp.int32)
    is_cube_close = 1.0 - jnp.tanh(20.0 * ee_cube_dist)
    is_close_goal = 1.0 - jnp.tanh(20.0 * cube_goal_dist)
    is_very_close = (1.0 - jnp.tanh(50.0 * ee_cube_dist)) * gripper_closed
    is_very_close_target = (1.0 - jnp.tanh(50.0 * cube_goal_dist)) * gripper_closed
    lift_reward = jnp.clip((current_cube_pos[..., 2] - 0.015) / 0.1, 0.0, 1.0)

    potential = (
        1.0 * is_cube_close
        + 2.0 * is_very_close
        + 4.0 * is_close_goal
        + 4.0 * is_very_close_target
        + 6.0 * lift_reward
    )

    action_delta = jnp.linalg.norm(prev_ctrl[..., :-1] - ctrl[..., :-1], axis=-1)

    # small bonus just for gripping, on top of the success bonus, to shape toward closing on the cube
    total_reward = potential + 2.0 * is_gripped + 10.0 * is_success - 0.01 * action_delta

    return (total_reward, is_touch, is_gripped, is_success)


def step_batch(
    cube_id, gripper_id, mjw_model, mjw_data, ctrl, goal_cube_pos, prev_ctrl,
    fixed_touch_adr, moving_touch_adr, n_frames=3,
):
    scaled_ctrl = scale_action_to_actuators(ctrl)
    ctrl_wp = wp.from_jax(scaled_ctrl)
    prev_ctrl = wp.to_jax(prev_ctrl)
    wp.copy(mjw_data.ctrl, ctrl_wp)

    for _ in range(n_frames):
        mjw.step(mjw_model, mjw_data)

    current_ee_pos = wp.to_jax(mjw_data.site_xpos)[:, gripper_id]
    current_cube_pos = wp.to_jax(mjw_data.xpos)[:, cube_id]

    fixed_force, moving_force = get_gripper_touch_forces(mjw_data, fixed_touch_adr, moving_touch_adr)
    
    return compute_rew(
            goal_cube_pos, current_cube_pos, current_ee_pos,
            scaled_ctrl, prev_ctrl, fixed_force, moving_force,
        )


if __name__ == '__main__':
    import numpy as np

    base_key = jax.random.key(12)
    k1, k2, k3 = jax.random.split(base_key, num=3)

    mj_model, mjw_model, mjw_data = init_mujoco(2)
    render_ctx, rgb_buff = init_rendering(mj_model, 2, (128, 128))

    cube_id = get_cube_id(mj_model)
    ee_id = get_gripper_id(mj_model)
    fixed_touch_adr = get_touch_sensor_adr(mj_model, "fixed_jaw_touch")
    moving_touch_adr = get_touch_sensor_adr(mj_model, "moving_jaw_touch")

    reset_batch(
        mj_model,
        mjw_model,
        mjw_data,
        k2,
        cube_id,
        pos_range=0.03,
        qpos_noise_scale=0.03
    )
    goal_cube_pos = find_goal_cube_pos(mj_model, mjw_data)

    frames = []
    prev_ctrl = wp.to_jax(mjw_data.ctrl)

    for i in range(100):
        k3, base_key = jax.random.split(base_key)
        obs = render_batch(mjw_model, mjw_data, render_ctx, rgb_buff)
        action = sample_action(mjw_data, k3)
        rew, is_touch, is_gripped, is_success = step_batch(
            cube_id, ee_id, mjw_model, mjw_data, action, goal_cube_pos, prev_ctrl,
            fixed_touch_adr, moving_touch_adr,
        )
        prev_ctrl = action
        frames.append(np.asarray(obs[0], dtype=np.uint8))

        
    media.write_video(path="vid.mp4", images=frames)