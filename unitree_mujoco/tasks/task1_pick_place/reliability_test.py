"""Run the task1 pick-and-place repeatedly and report how often it succeeds.

A trial succeeds when the controller reaches "done" with the block resting on
the green pad. Each trial gets a fresh MjData and a fresh controller, and the
block's start pose is jittered a little so the run is a real reliability check
rather than the same deterministic replay ten times over.

    python reliability_test.py [trials] [--seconds N] [--jitter M] [--quiet]
"""

import contextlib
import io
import sys

import mujoco
import numpy as np

from task1_hand_controller import Task1HandController

MODEL = "task1.xml"
DEFAULT_TRIALS = 10
DEFAULT_SECONDS = 70.0
DEFAULT_JITTER = 0.010
CONTROL_DT = 0.005


def run_trial(model, seconds, jitter, rng, verbose):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cube = model.body("red_cube").id
    qadr = model.jnt_qposadr[model.body_jntadr[cube]]
    if jitter > 0.0:
        data.qpos[qadr] += rng.uniform(-jitter, jitter)
        data.qpos[qadr + 1] += rng.uniform(-jitter, jitter)
        yaw = rng.uniform(-np.pi / 12.0, np.pi / 12.0)
        data.qpos[qadr + 3] = np.cos(yaw / 2.0)
        data.qpos[qadr + 6] = np.sin(yaw / 2.0)
    mujoco.mj_forward(model, data)
    start = data.xpos[cube].copy()

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink if not verbose else sys.stdout):
        controller = Task1HandController(model, data)

    model.opt.timestep = CONTROL_DT
    steps = int(seconds / CONTROL_DT)
    grasped = False

    for _ in range(steps):
        with contextlib.redirect_stdout(sink if not verbose else sys.stdout):
            controller.step()
        mujoco.mj_step(model, data)
        if controller.holding:
            grasped = True
        if controller.phase == "done":
            break

    placed = controller._cube_on_pad()
    return {
        "placed": bool(placed),
        "grasped": grasped,
        "phase": controller.phase,
        "attempts": controller.attempts,
        "time": float(data.time),
        "start": start,
        "end": data.xpos[cube].copy(),
        "log": sink.getvalue(),
    }


def main():
    args = [a for a in sys.argv[1:]]
    trials = DEFAULT_TRIALS
    seconds = DEFAULT_SECONDS
    jitter = DEFAULT_JITTER
    verbose = False

    rest = []
    while args:
        item = args.pop(0)
        if item == "--seconds":
            seconds = float(args.pop(0))
        elif item == "--jitter":
            jitter = float(args.pop(0))
        elif item in ("--verbose", "-v"):
            verbose = True
        else:
            rest.append(item)
    if rest:
        trials = int(rest[0])

    model = mujoco.MjModel.from_xml_path(MODEL)
    rng = np.random.default_rng(0)

    print(
        f"running {trials} trial(s), {seconds:.0f} s each, "
        f"block jitter +/-{jitter * 100:.1f} cm and +/-15 deg yaw"
    )
    print()

    successes = 0
    streak = 0
    for index in range(1, trials + 1):
        result = run_trial(model, seconds, jitter, rng, verbose)
        ok = result["placed"]
        successes += ok
        streak = streak + 1 if ok else 0
        offset = result["start"][:2] - np.array([0.34, 0.26])
        print(
            f"  trial {index:2d}  {'PASS' if ok else 'FAIL'}  "
            f"phase={result['phase']:12s} retries={result['attempts']} "
            f"t={result['time']:5.1f}s  "
            f"start_offset=({offset[0]:+.3f},{offset[1]:+.3f}) "
            f"block_end={np.round(result['end'], 3)}"
        )
        if not ok:
            tail = [
                line
                for line in result["log"].splitlines()
                if "->" in line or "recover" in line or "giving up" in line
            ]
            for line in tail[-8:]:
                print(f"        {line}")

    print()
    print(f"{successes}/{trials} placed on the pad")
    if successes == trials:
        print(f"PASS: {trials} consecutive successful pick-and-place cycles")
    else:
        print(f"FAIL: longest trailing streak {streak}")
    return 0 if successes == trials else 1


if __name__ == "__main__":
    sys.exit(main())
