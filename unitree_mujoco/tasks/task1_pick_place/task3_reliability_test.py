"""Run Task 3 pick-rotate-push-plant until 10 consecutive successes.

A trial succeeds only when all of these hold at the end:
  1. The cube's four corners rest on the green pad
  2. The stick stands on the cube (vertical, on top, centered)
  3. The hand is open and no longer holding the stick
  4. The arm has withdrawn (grasp site clear of the stick)
  5. The controller finished in phase "done" after using the stick to push

    python task3_reliability_test.py [trials] [--seconds N] [--verbose]
    python task3_reliability_test.py --until-streak 10 [--max-trials N]
"""

import contextlib
import io
import os
import sys

import mujoco
import numpy as np

from task3_hand_controller import Task3HandController

SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task3.xml")
DEFAULT_TRIALS = 10
DEFAULT_SECONDS = 70.0
DEFAULT_MAX_TRIALS = 40
CONTROL_DT = 0.005
ARM_CLEAR_XY = 0.08


def run_trial(model, seconds, verbose):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    sink = io.StringIO()
    controller = None
    try:
        with contextlib.redirect_stdout(sink if not verbose else sys.stdout):
            controller = Task3HandController(model, data)
        model.opt.timestep = CONTROL_DT

        steps = int(seconds / CONTROL_DT)
        for _ in range(steps):
            with contextlib.redirect_stdout(sink if not verbose else sys.stdout):
                controller.step()
            mujoco.mj_step(model, data)
            if controller.phase == "done" and controller.phase_time > 0.6:
                break
    finally:
        if controller is not None:
            controller._set_hand_collisions(True)

    stick = data.xpos[model.body("tool_stick").id].copy()
    cube = data.xpos[model.body("obj_cube").id].copy()
    pad = data.xpos[model.body("green_target").id].copy()
    site = data.site_xpos[controller.site_id].copy()
    axis_z = float(controller._stick_axis()[2])
    on_pad = bool(controller._cube_on_pad())
    standing = bool(controller._stick_standing_on_cube())
    held = bool(controller._stick_held())
    hand_hit = bool(controller._hand_hits_stick())
    arm_clear = float(np.linalg.norm(site[:2] - stick[:2])) > ARM_CLEAR_XY
    ok = bool(
        controller.succeeded
        and controller.phase == "done"
        and on_pad
        and standing
        and not held
        and not hand_hit
        and arm_clear
    )
    return {
        "ok": ok,
        "succeeded": bool(controller.succeeded),
        "on_pad": on_pad,
        "standing": standing,
        "held": held,
        "hand_hit": hand_hit,
        "arm_clear": arm_clear,
        "phase": controller.phase,
        "attempts": controller.attempts,
        "time": float(data.time),
        "stick": stick,
        "cube": cube,
        "pad": pad,
        "site": site,
        "axis_z": axis_z,
        "xy_mm": float(np.linalg.norm(stick[:2] - cube[:2]) * 1000.0),
        "log": sink.getvalue(),
    }


def _fail_tail(log):
    lines = [
        line
        for line in log.splitlines()
        if "->" in line
        or "recover" in line
        or "success" in line
        or "giving up" in line
        or "opening" in line
        or "withdraw" in line
        or "standing" in line
        or "timed out" in line
        or "dropped" in line
    ]
    return lines[-16:]


def _print_result(index, result):
    print(
        f"  trial {index:2d}  {'PASS' if result['ok'] else 'FAIL'}  "
        f"phase={result['phase']:12s} retries={result['attempts']} "
        f"t={result['time']:5.1f}s  "
        f"pad={int(result['on_pad'])} stand={int(result['standing'])} "
        f"held={int(result['held'])} clear={int(result['arm_clear'])} "
        f"az={result['axis_z']:+.3f} xy={result['xy_mm']:5.0f}mm"
    )
    print(
        f"           stick={np.round(result['stick'], 3)} "
        f"cube={np.round(result['cube'], 3)} "
        f"site={np.round(result['site'], 3)}"
    )
    if not result["ok"]:
        for line in _fail_tail(result["log"]):
            print(f"        {line}")


def main():
    args = list(sys.argv[1:])
    trials = DEFAULT_TRIALS
    seconds = DEFAULT_SECONDS
    verbose = False
    until_streak = None
    max_trials = DEFAULT_MAX_TRIALS
    rest = []
    while args:
        item = args.pop(0)
        if item == "--seconds":
            seconds = float(args.pop(0))
        elif item in ("--verbose", "-v"):
            verbose = True
        elif item == "--until-streak":
            until_streak = int(args.pop(0))
        elif item == "--max-trials":
            max_trials = int(args.pop(0))
        else:
            rest.append(item)
    if rest:
        trials = int(rest[0])
    if until_streak is None:
        until_streak = trials
        max_trials = trials

    model = mujoco.MjModel.from_xml_path(SCENE)
    print(
        f"running Task 3 until {until_streak} consecutive successes "
        f"(max {max_trials} trials, {seconds:.0f} s each)"
    )
    print()

    successes = 0
    streak = 0
    index = 0
    while index < max_trials and streak < until_streak:
        index += 1
        result = run_trial(model, seconds, verbose)
        successes += int(result["ok"])
        streak = streak + 1 if result["ok"] else 0
        _print_result(index, result)

    print()
    print(
        f"{successes}/{index} succeeded, "
        f"best consecutive streak {streak}"
    )
    if streak >= until_streak:
        print(f"PASS: {until_streak} consecutive successful Task 3 cycles")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
