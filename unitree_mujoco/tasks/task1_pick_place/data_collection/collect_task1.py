"""Collect Task 1 VLA episodes (success + flagged boundary failures).

    python collect_task1.py --out datasets/task1/smoke --n 10 --seed 0
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import mujoco
import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

from task1_hand_controller import Task1HandController

from data_collection.boundary_policy import BoundaryPolicy
from data_collection.randomize import randomize_episode
from data_collection.recorder import (
    EpisodeRecorder,
    action_indices,
    compute_action,
    fingers_closing,
    hand_hits_cube,
)
from data_collection.rewards import RewardTracker
from data_collection.schema import (
    CONTROL_DT,
    RECORD_HZ,
    SMOKE_KINDS,
    instruction_for,
)


SCENE = os.path.join(TASK_DIR, "task1_collect.xml")


def _kind_for(index, n, kinds):
    if n == len(SMOKE_KINDS) and not kinds:
        return SMOKE_KINDS[index]
    if kinds:
        return kinds[index % len(kinds)]
    if index < max(1, int(round(0.5 * n))):
        return "success"
    if index < max(2, int(round(0.8 * n))):
        return "space_constraint"
    return "kinematics_limit"


def _snapshot(controller, action, phase, phase_time):
    cube = controller.data.xpos[controller.cube_id]
    pad = controller.data.xpos[controller.target_id]
    site = controller.data.site_xpos[controller.site_id]
    return {
        "ee_pos": site.copy(),
        "cube_pos": cube.copy(),
        "pad_pos": pad.copy(),
        "cube_vel": controller.data.qvel[
            controller.cube_vadr : controller.cube_vadr + 3
        ].copy(),
        "phase": phase,
        "phase_time": float(phase_time),
        "holding": bool(controller.holding),
        "grip_firm": bool(controller._grip_is_firm()),
        "fingers_closing": fingers_closing(controller),
        "hand_hits_cube": hand_hits_cube(controller),
        "cube_on_pad": bool(controller._cube_on_pad()),
        "cube_tipped": bool(controller._cube_tipped()),
        "action": action,
    }


def run_episode(model, kind, rng, seconds, verbose):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    variant = randomize_episode(model, data, kind, rng)
    instruction, paraphrase_id = instruction_for(kind, rng)

    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink if not verbose else sys.stdout)
    with ctx:
        controller = Task1HandController(model, data)
    model.opt.timestep = CONTROL_DT

    if kind == "success":
        policy = None
        action_mode = "delta_q"
    else:
        with ctx:
            policy = BoundaryPolicy(controller, kind)
        action_mode = policy.action_mode

    act_ids, act_qadr = action_indices(model, controller.act_of_joint)
    rewards = RewardTracker(
        controller.table_top, controller.cube_half, controller.pad_half
    )
    recorder = EpisodeRecorder(model)

    steps = int(seconds / CONTROL_DT)
    for _ in range(steps):
        with ctx:
            if policy is None:
                controller.step()
                phase = controller.phase
                phase_time = controller.phase_time
            else:
                policy.step(CONTROL_DT)
                phase = policy.phase
                phase_time = policy.phase_time
                controller.phase = phase
        mujoco.mj_step(model, data)

        action = compute_action(controller, act_ids, act_qadr)
        if kind == "kinematics_limit":
            action = np.zeros_like(action)
            controller.q_target[act_ids] = controller.data.qpos[act_qadr]

        if recorder.should_record():
            snap = _snapshot(controller, action, phase, phase_time)
            reward, terms, events = rewards.step(snap, kind)
            recorder.maybe_record(
                data, controller, action, phase, reward, terms, events
            )
        else:
            recorder.tick += 1

        if policy is None:
            if controller.phase == "done" and controller.phase_time > 0.4:
                break
        elif policy.done:
            break

    placed = bool(controller._cube_on_pad())
    if kind == "success":
        is_success = placed and controller.phase == "done"
        failure_reason = None if is_success else "space_constraint"
    else:
        is_success = False
        failure_reason = kind

    meta = {
        "episode_kind": kind,
        "instruction": instruction,
        "paraphrase_id": paraphrase_id,
        "is_success": bool(is_success),
        "failure_reason": failure_reason,
        "action_mode": action_mode,
        "fps": RECORD_HZ,
        "sim_time": float(data.time),
        "controller_phase": controller.phase,
        "placed": placed,
        **variant,
    }
    return recorder, meta, sink.getvalue()


def _validate_smoke(rows):
    errors = []
    for row in rows:
        if row["num_steps"] < 1:
            errors.append(f"{row['episode_id']} has no steps")
        if row["episode_kind"] == "success" and not row["is_success"]:
            errors.append(f"{row['episode_id']} success kind but is_success=false")
        if row["episode_kind"] != "success":
            if row["is_success"]:
                errors.append(f"{row['episode_id']} failure kind marked success")
            if not row["failure_reason"]:
                errors.append(f"{row['episode_id']} missing failure_reason")
        if row["num_steps"] != row["_t_phase"]:
            errors.append(f"{row['episode_id']} phase/reward length mismatch")
        if row["episode_kind"] == "success" and row["is_success"]:
            terms = row.get("phase_returns") or {}
            if row["return"] <= 0.0:
                errors.append(f"{row['episode_id']} success return is not positive")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Collect Task 1 VLA episodes")
    parser.add_argument("--out", default="datasets/task1/smoke")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=70.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--kinds",
        default="",
        help="Comma list of kinds to cycle (default: smoke mix if n=10)",
    )
    args = parser.parse_args()

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    out_root = args.out
    if not os.path.isabs(out_root):
        out_root = os.path.join(TASK_DIR, out_root)
    os.makedirs(out_root, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    model = mujoco.MjModel.from_xml_path(SCENE)

    print(
        f"collecting {args.n} Task 1 episode(s) to {out_root} "
        f"(seed={args.seed}, {args.seconds:.0f}s cap)",
        flush=True,
    )
    rows = []
    for index in range(args.n):
        kind = _kind_for(index, args.n, kinds)
        # Fresh model so visual mutations do not leak across episodes.
        model = mujoco.MjModel.from_xml_path(SCENE)
        recorder, meta, _log = run_episode(
            model, kind, rng, args.seconds, args.verbose
        )
        if kind == "success" and not meta["is_success"]:
            for retry in range(2):
                model = mujoco.MjModel.from_xml_path(SCENE)
                recorder, meta, _log = run_episode(
                    model, kind, rng, args.seconds, args.verbose
                )
                if meta["is_success"]:
                    break
        episode_id = f"ep_{index:06d}"
        ep_dir = os.path.join(out_root, episode_id)
        payload = recorder.write(ep_dir, episode_id, meta)
        payload["_t_phase"] = recorder.num_steps
        rows.append(payload)
        print(
            f"  {episode_id}  {kind:18s}  "
            f"{'PASS' if payload['is_success'] else 'FAIL':4s}  "
            f"reason={payload['failure_reason']}  "
            f"T={payload['num_steps']:4d}  R={payload['return']:+6.2f}  "
            f"finish={payload['cube_finish']}",
            flush=True,
        )

    errors = _validate_smoke(rows)
    print()
    print(f"wrote {len(rows)} episodes")
    if errors:
        print("SMOKE CHECKS FAILED:")
        for err in errors:
            print(f"  {err}")
        return 1
    print("SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
