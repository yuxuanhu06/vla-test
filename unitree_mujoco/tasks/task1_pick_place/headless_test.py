"""Run the Task1 hand controller without a viewer or the DDS bridge.

Usage: python headless_test.py [seconds]
"""

import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task1_hand_controller import Task1HandController

SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task1.xml")


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    controller = Task1HandController(model, data)
    model.opt.timestep = 0.005

    steps = int(duration / model.opt.timestep)
    started = time.perf_counter()
    max_qacc = 0.0

    for _ in range(steps):
        controller.step()
        mujoco.mj_step(model, data)
        max_qacc = max(max_qacc, float(np.abs(data.qacc).max()))
        if controller.phase == "done":
            break

    cube = data.xpos[model.body("red_cube").id]
    pad = data.xpos[model.body("green_target").id]
    offset = np.linalg.norm(cube[:2] - pad[:2])

    print()
    print("-" * 62)
    print(f"wall time        : {time.perf_counter() - started:.1f} s "
          f"for {data.time:.1f} s simulated")
    print(f"final phase      : {controller.phase}")
    print(f"attempts used    : {controller.attempts}")
    print(f"cube             : {np.round(cube, 4)}")
    print(f"pad              : {np.round(pad, 4)}")
    print(f"xy offset to pad : {offset:.4f} m (pad half-extent "
          f"{controller.pad_half:.3f})")
    print(f"cube on pad      : {controller._cube_on_pad()}")
    print(f"max |qacc|       : {max_qacc:.1f}")
    print("-" * 62)
    print("RESULT:", "SUCCESS" if controller._cube_on_pad() else "FAILURE")


if __name__ == "__main__":
    main()
