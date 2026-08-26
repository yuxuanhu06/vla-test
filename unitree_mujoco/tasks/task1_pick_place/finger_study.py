"""Compare finger-joint schedules for the Dex3 grasp on the task1 block.

The left hand has seven driven joints: thumb_0/1/2, index_0/1 and middle_0/1.
This script parks the arm on the grasp waypoint, closes a chosen subset of
those joints to chosen angles, then lifts the arm and reports whether the block
came with it. Grip force alone is misleading, so the verdict is the lift.

    python finger_study.py
"""

import contextlib
import io
import sys

import mujoco
import numpy as np

from task1_hand_controller import Task1HandController

MODEL = "task1.xml"
DT = 0.005

THUMB = ("left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
         "left_hand_thumb_2_joint")
INDEX = ("left_hand_index_0_joint", "left_hand_index_1_joint")
MIDDLE = ("left_hand_middle_0_joint", "left_hand_middle_1_joint")


def schedules():
    """Named finger targets. Missing joints stay open."""
    base = {
        "left_hand_thumb_0_joint": 0.55,
        "left_hand_thumb_1_joint": 0.85,
        "left_hand_thumb_2_joint": 1.40,
        "left_hand_index_0_joint": -0.95,
        "left_hand_index_1_joint": -1.45,
        "left_hand_middle_0_joint": -0.95,
        "left_hand_middle_1_joint": -1.45,
    }
    pick = lambda names: {k: v for k, v in base.items() if k in names}

    out = {
        "all 7 joints": dict(base),
        "thumb + index": pick(THUMB + INDEX),
        "thumb + middle": pick(THUMB + MIDDLE),
        "index + middle (no thumb)": pick(INDEX + MIDDLE),
        "thumb only": pick(THUMB),
    }

    # Thumb opposition: thumb_0 swings the thumb across the jaw, so it decides
    # whether the pad meets the block face or slides past it.
    for t0 in (0.25, 0.40, 0.70, 0.90):
        s = dict(base)
        s["left_hand_thumb_0_joint"] = t0
        out[f"all 7, thumb_0={t0:.2f}"] = s

    # Proximal-led wrap: knuckles take the block, tips only conform.
    s = dict(base)
    s.update({"left_hand_index_1_joint": -0.80,
              "left_hand_middle_1_joint": -0.80,
              "left_hand_thumb_2_joint": 1.00})
    out["proximal-led wrap"] = s

    # Distal-led pinch: fingertips do the work.
    s = dict(base)
    s.update({"left_hand_index_0_joint": -0.55,
              "left_hand_middle_0_joint": -0.55,
              "left_hand_index_1_joint": -1.70,
              "left_hand_middle_1_joint": -1.70,
              "left_hand_thumb_2_joint": 1.70})
    out["distal-led pinch"] = s

    return out


def hold(controller, data, model, seconds, closing):
    for _ in range(int(seconds / DT)):
        controller._slew_to_ik(DT)
        controller._apply_pd(closing_fingers=closing)
        mujoco.mj_step(model, data)


def trial(model, targets, verbose=False):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        c = Task1HandController(model, data)
    model.opt.timestep = DT

    block = data.xpos[c.cube_id].copy()
    grasp = c._grasp_point(block)

    # Park the arm on the grasp waypoint. Clearance there is positive, so
    # teleporting in does not disturb the block.
    with contextlib.redirect_stdout(sink):
        q, residual, _ = c._solve_ik(grasp)
    if residual > 0.02:
        return None
    data.qpos[c.ik_qadr] = q
    data.qvel[c.ik_dofs] = 0.0
    c.q_ik = q.copy()
    for act, value in zip(c.ik_acts, q):
        c.q_target[act] = value
    mujoco.mj_forward(model, data)

    c._set_fingers_open()
    hold(c, data, model, 0.4, closing=False)

    for act in c.finger_acts:
        c.q_target[act] = c.finger_open_target[act]
    for name, value in targets.items():
        c.q_target[c.act_of_joint[name]] = value
    hold(c, data, model, 1.6, closing=True)

    thumb, fingers, force = c._grip_state()
    reached = {
        name: float(data.qpos[c.qadr[c.act_of_joint[name]]])
        for name in targets
    }
    z_before = float(data.xpos[c.cube_id][2])

    with contextlib.redirect_stdout(sink):
        c.q_ik = c._solve_ik(c._at_clearance(grasp))[0]
    hold(c, data, model, 3.5, closing=True)

    z_after = float(data.xpos[c.cube_id][2])
    lifted = z_after - z_before
    still_held = c._grip_state()[2] > 0.5
    return {
        "thumb": thumb,
        "fingers": fingers,
        "force": force,
        "lift": lifted,
        "held": still_held,
        "reached": reached,
        "targets": targets,
    }


def main():
    model = mujoco.MjModel.from_xml_path(MODEL)
    print("Parking the arm on the grasp waypoint, closing the listed joints,")
    print("then lifting 0.10 m. 'lift' is how far the block actually rose.")
    print()
    header = (f'{"schedule":28s} {"thumb":>5s} {"fing":>4s} {"force":>7s} '
              f'{"lift":>7s}  verdict')
    print(header)
    print("-" * len(header))

    results = []
    for name, targets in schedules().items():
        r = trial(model, targets)
        if r is None:
            print(f"{name:28s}  grasp pose unreachable")
            continue
        ok = r["lift"] > 0.05 and r["held"]
        verdict = "HOLDS" if ok else ("slipped" if r["lift"] > 0.01 else "dropped")
        print(f'{name:28s} {r["thumb"]:5d} {r["fingers"]:4d} '
              f'{r["force"]:7.2f} {r["lift"]:+7.3f}  {verdict}')
        results.append((name, r, ok))

    print()
    winners = [(n, r) for n, r, ok in results if ok]
    if not winners:
        print("nothing held the block through the lift")
        best = max(results, key=lambda x: x[1]["lift"], default=None)
        if best:
            print(f'closest: {best[0]} lifted it {best[1]["lift"]:+.3f} m')
        return 1

    best = max(winners, key=lambda x: x[1]["force"])
    print(f'best hold: {best[0]}  (force {best[1]["force"]:.2f} N, '
          f'lift {best[1]["lift"]:+.3f} m)')
    print("joint angles it actually reached versus commanded:")
    for name, value in best[1]["reached"].items():
        print(f'  {name:32s} reached {value:+.3f} '
              f'commanded {best[1]["targets"][name]:+.3f}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
