"""Run Task 2 pick-and-place until 10 consecutive successes, or report failures.

A trial succeeds only when:
  1. FINAL DECISION matches the forced object type
  2. That body rests on the green pad
  3. The other three objects have not been knocked (XY move < 2 cm, still upright)

    python task2_reliability_test.py [trials] [--seconds N] [--verbose]
"""

import contextlib
import io
import os
import random
import sys

import mujoco
import numpy as np

from task2_hand_controller import Task2HandController

SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task2.xml")
OBJECT_CYCLE = ("tri_prism", "cylinder", "cube", "rect_prism")
NEIGHBOR_MOVE_TOL = 0.02
DEFAULT_TRIALS = 10
DEFAULT_SECONDS = 70.0
CONTROL_DT = 0.005


def _neighbor_state(model, data, skip_type):
    poses = {}
    upright = {}
    for typ, (body_name, _geom) in Task2HandController.OBJECT_TYPES.items():
        if typ == skip_type:
            continue
        body_id = model.body(body_name).id
        poses[typ] = data.xpos[body_id][:2].copy()
        up = data.xmat[body_id].reshape(3, 3)[:, 2]
        upright[typ] = bool(up[2] >= 0.8)
    return poses, upright


def run_trial(model, seconds, forced_type, verbose):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    original_choice = random.choice

    def _forced_choice(seq):
        items = list(seq)
        if forced_type in items:
            return forced_type
        return original_choice(items)

    random.choice = _forced_choice
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink if not verbose else sys.stdout):
            controller = Task2HandController(model, data)
        model.opt.timestep = CONTROL_DT

        neighbor_start = None
        decision = None
        pick_pos = None
        steps = int(seconds / CONTROL_DT)

        for _ in range(steps):
            with contextlib.redirect_stdout(sink if not verbose else sys.stdout):
                controller.step()
            mujoco.mj_step(model, data)

            if (
                neighbor_start is None
                and controller.required_type is not None
                and controller.cube_id is not None
            ):
                decision = controller.required_type
                pick_pos = data.xpos[controller.cube_id].copy()
                neighbor_start, _ = _neighbor_state(
                    model, data, controller.required_type
                )

            if controller.phase == "done":
                break
    finally:
        random.choice = original_choice

    log = sink.getvalue()
    pad = data.xpos[model.body("green_target").id].copy()
    placed = False
    end = None
    if controller.cube_id is not None:
        end = data.xpos[controller.cube_id].copy()
        placed = bool(controller._cube_on_pad())

    neighbor_end, neighbor_up = _neighbor_state(
        model, data, decision or forced_type
    )
    neighbor_deltas = {}
    neighbors_ok = True
    if neighbor_start is not None:
        for typ, start_xy in neighbor_start.items():
            delta = float(np.linalg.norm(neighbor_end[typ] - start_xy))
            neighbor_deltas[typ] = delta
            if delta > NEIGHBOR_MOVE_TOL or not neighbor_up.get(typ, False):
                neighbors_ok = False
    else:
        neighbors_ok = False

    decision_ok = decision == forced_type
    ok = bool(
        decision_ok
        and placed
        and controller.phase == "done"
        and neighbors_ok
    )
    return {
        "ok": ok,
        "decision_ok": decision_ok,
        "placed": placed,
        "neighbors_ok": neighbors_ok,
        "forced": forced_type,
        "decision": decision,
        "phase": controller.phase,
        "attempts": controller.attempts,
        "time": float(data.time),
        "pick": pick_pos,
        "end": end,
        "pad": pad,
        "neighbor_deltas": neighbor_deltas,
        "log": log,
    }


def _fail_tail(log):
    lines = [
        line
        for line in log.splitlines()
        if "->" in line
        or "recover" in line
        or "FINAL DECISION" in line
        or "giving up" in line
        or "centred" in line
        or "lowering" in line
        or "UNREACHABLE" in line
    ]
    return lines[-12:]


def main():
    args = list(sys.argv[1:])
    trials = DEFAULT_TRIALS
    seconds = DEFAULT_SECONDS
    verbose = False
    rest = []
    while args:
        item = args.pop(0)
        if item == "--seconds":
            seconds = float(args.pop(0))
        elif item in ("--verbose", "-v"):
            verbose = True
        else:
            rest.append(item)
    if rest:
        trials = int(rest[0])

    model = mujoco.MjModel.from_xml_path(SCENE)
    print(
        f"running {trials} Task 2 trial(s), {seconds:.0f} s each, "
        f"cycling {OBJECT_CYCLE}"
    )
    print()

    successes = 0
    for index in range(1, trials + 1):
        forced = OBJECT_CYCLE[(index - 1) % len(OBJECT_CYCLE)]
        result = run_trial(model, seconds, forced, verbose)
        ok = result["ok"]
        successes += int(ok)
        end = np.round(result["end"], 3) if result["end"] is not None else None
        pick = np.round(result["pick"], 3) if result["pick"] is not None else None
        deltas = {
            k: f"{v * 1000:.0f}mm" for k, v in result["neighbor_deltas"].items()
        }
        print(
            f"  trial {index:2d}  {'PASS' if ok else 'FAIL'}  "
            f"type={result['forced']:11s} "
            f"decision={result['decision']}  "
            f"phase={result['phase']:17s} retries={result['attempts']} "
            f"t={result['time']:5.1f}s"
        )
        print(
            f"           pick={pick} end={end} pad={np.round(result['pad'], 3)} "
            f"placed={result['placed']} neighbors={deltas}"
        )
        if not ok:
            for line in _fail_tail(result["log"]):
                print(f"        {line}")
            if os.environ.get("TASK2_DUMP_LOG"):
                print("----- full log -----")
                print(result["log"])

    print()
    print(f"{successes}/{trials} placed on the pad without disturbing neighbors")
    if successes == trials:
        print(f"PASS: {trials} consecutive successful Task 2 cycles")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
