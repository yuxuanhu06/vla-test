import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import sys
import os

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config


# ---------------------------------------------------------
# Task selection: argv overrides config.TASK
# mjpython ./unitree_mujoco.py task1
# mjpython ./unitree_mujoco.py task2
# mjpython ./unitree_mujoco.py task3
# ---------------------------------------------------------

TASK = sys.argv[1] if len(sys.argv) > 1 else config.TASK
if TASK not in ("task1", "task2", "task3"):
    print("Usage: mjpython ./unitree_mujoco.py [task1|task2|task3]")
    sys.exit(1)

config.TASK = TASK
config.ROBOT_SCENE = "../tasks/task1_pick_place/" + TASK + ".xml"
print("Running", config.TASK)


# ---------------------------------------------------------
# Import task controller
# ---------------------------------------------------------

TASK_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../tasks/task1_pick_place",
    )
)

if TASK_DIR not in sys.path:
    sys.path.append(TASK_DIR)

if config.TASK == "task3":
    from task3_hand_controller import Task3HandController as TaskController
elif config.TASK == "task2":
    from task2_hand_controller import Task2HandController as TaskController
else:
    from task1_hand_controller import Task1HandController as TaskController


# ---------------------------------------------------------
# Global lock
# ---------------------------------------------------------

locker = threading.Lock()


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

mujoco.mj_forward(mj_model, mj_data)

print("Loaded scene:", config.ROBOT_SCENE)
print("Number of actuators:", mj_model.nu)


# ---------------------------------------------------------
# Task controller
# ---------------------------------------------------------

task_controller = TaskController(
    mj_model,
    mj_data,
)


# ---------------------------------------------------------
# Elastic band + viewer
# ---------------------------------------------------------

if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()

    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id

    viewer = mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
        key_callback=elastic_band.MujuocoKeyCallback,
    )

else:
    viewer = mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
    )


# ---------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------

mj_model.opt.timestep = config.SIMULATE_DT

num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


# ---------------------------------------------------------
# Simulation thread
# ---------------------------------------------------------

def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(
        config.DOMAIN_ID,
        config.INTERFACE,
    )

    unitree = UnitreeSdk2Bridge(
        mj_model,
        mj_data,
    )

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(
            device_id=config.JOYSTICK_DEVICE,
            js_type=config.JOYSTICK_TYPE,
        )

    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():

        step_start = time.perf_counter()

        locker.acquire()

        try:

            # ---------------------------------------------
            # Elastic band
            # ---------------------------------------------

            if config.ENABLE_ELASTIC_BAND:
                if elastic_band.enable:

                    mj_data.xfrc_applied[
                        band_attached_link,
                        :3,
                    ] = elastic_band.Advance(
                        mj_data.qpos[:3],
                        mj_data.qvel[:3],
                    )

            # ---------------------------------------------
            # Task controller
            # IMPORTANT:
            # write controls BEFORE mj_step()
            # ---------------------------------------------

            task_controller.step()

            # ---------------------------------------------
            # MuJoCo physics
            # ---------------------------------------------

            mujoco.mj_step(
                mj_model,
                mj_data,
            )

        finally:
            locker.release()

        # ---------------------------------------------
        # Maintain simulation timestep
        # ---------------------------------------------

        time_until_next_step = (
            mj_model.opt.timestep
            - (time.perf_counter() - step_start)
        )

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


# ---------------------------------------------------------
# Viewer thread
# ---------------------------------------------------------

def PhysicsViewerThread():

    while viewer.is_running():

        locker.acquire()

        try:
            viewer.sync()

        finally:
            locker.release()

        time.sleep(config.VIEWER_DT)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    viewer_thread = Thread(
        target=PhysicsViewerThread,
        daemon=True,
    )

    sim_thread = Thread(
        target=SimulationThread,
        daemon=True,
    )

    viewer_thread.start()
    sim_thread.start()

    viewer_thread.join()
    sim_thread.join()
