
import os
import time

import numpy as np
import mujoco
import mujoco.viewer


def get_cube_id(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "redcube")


def get_gripper_id(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")


def get_touch_sensor_adr(model, name):
    """Address of a scalar touch sensor's value in sensordata."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    assert sid >= 0, f"sensor '{name}' not found in model"
    return int(model.sensor_adr[sid])


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(curr_dir, "env/scene.xml")
    print(f"Loading MJCF: {scene_path}")

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "grasp_ready")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print(f"Starting from keyframe 'grasp_ready' (key_id={key_id})")
    else:
        mujoco.mj_resetData(model, data)
        print("No 'grasp_ready' keyframe found, starting from qpos0")
    mujoco.mj_forward(model, data)

    cube_id = get_cube_id(model)
    gripper_site_id = get_gripper_id(model)
    fixed_adr = get_touch_sensor_adr(model, "fixed_jaw_touch")
    moving_adr = get_touch_sensor_adr(model, "moving_jaw_touch")

    last_print = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            now = time.time()
            if now - last_print > 0.5:
                ee_pos = data.site_xpos[gripper_site_id]
                cube_pos = data.xpos[cube_id]
                dist = float(np.linalg.norm(ee_pos - cube_pos))

                fixed_force = float(data.sensordata[fixed_adr])
                moving_force = float(data.sensordata[moving_adr])

                is_touch = dist < 0.017
                is_gripped = (fixed_force > 0.4) and (moving_force > 0.4) and is_touch

                print(
                    f"ee<->cube dist: {dist:.4f} m | "
                    f"fixed_force: {fixed_force:6.3f} N | "
                    f"moving_force: {moving_force:6.3f} N | "
                    f"cube_z: {cube_pos[2]:.4f} | "
                    f"is_touch(<0.017): {is_touch} | "
                    f"is_gripped(>0.4N both): {is_gripped}"
                )
                last_print = now

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()