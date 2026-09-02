"""Collect Task 3 VLA episodes (success + flagged boundary failures).

    python collect_task3.py --out datasets/task3/smoke --n 10 --seed 0
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

from task3_hand_controller import Task3HandController

from data_collection.boundary_policy_task3 import BoundaryPolicy
from data_collection.libero_writer import LiberoPackWriter, demo_gripper_is_left
from data_collection.randomize_task3 import randomize_episode
from data_collection.recorder import (
    EpisodeRecorder,
    action_indices,
    compute_action,
    fingers_closing,
)
from data_collection.rewards_task3 import RewardTracker
from data_collection.schema_task3 import (
    CANONICAL_INSTRUCTION,
    CONTROL_DT,
    RECORD_HZ,
    REWARD_TERMS,
    SMOKE_SPECS,
    TASK_ID,
    instruction_for,
)


SCENE = os.path.join(TASK_DIR, "task3_collect.xml")
ARM_CLEAR_XY = 0.08
EXPERT_FAIL_KINDS = ("swapped", "no_cube", "cube_too_small", "stick_too_short")


def _kind_for(index, n, kinds):
    if n == len(SMOKE_SPECS) and not kinds:
        return SMOKE_SPECS[index][0], SMOKE_SPECS[index][1]
    if kinds:
        return kinds[index % len(kinds)], None
    success_n = max(1, int(round(0.30 * n)))
    order = [
        (success_n, "success"),
        (max(1, int(round(0.10 * n))), "swapped"),
        (max(1, int(round(0.10 * n))), "too_far"),
        (max(1, int(round(0.10 * n))), "no_cube"),
        (max(1, int(round(0.15 * n))), "space_constraint"),
        (max(1, int(round(0.10 * n))), "cube_too_small"),
        (max(1, int(round(0.10 * n))), "stick_too_short"),
    ]
    cursor = 0
    for count, kind in order:
        if index < cursor + count:
            return kind, None
        cursor += count
    return "stick_lying", None


def _object_vel(controller, vadr):
    return controller.data.qvel[vadr : vadr + 3].copy()


def _snapshot(controller, action, phase, phase_time):
    stick = controller.data.xpos[controller.stick_id].copy()
    cube = controller.data.xpos[controller.cube_id].copy()
    pad = controller.data.xpos[controller.target_id].copy()
    site = controller.data.site_xpos[controller.site_id].copy()
    axis_z = float(controller._stick_axis()[2])
    return {
        "ee_pos": site,
        "stick_pos": stick,
        "cube_pos": cube,
        "pad_pos": pad,
        "cube_vel": _object_vel(controller, controller.cube_vadr),
        "phase": phase,
        "phase_time": float(phase_time),
        "holding": bool(controller._stick_held()),
        "grip_firm": bool(controller._stick_held()),
        "fingers_closing": fingers_closing(controller),
        "hand_hits_cube": bool(controller._hand_hits_cube()),
        "cube_on_pad": bool(controller._cube_on_pad()),
        "stick_standing_on_cube": bool(controller._stick_standing_on_cube()),
        "stick_lying": bool(not controller._stick_standing_on_table() and axis_z < 0.8),
        "action": action,
        "rotated_in_air": bool(controller.rotated_in_air),
        "face_contact": bool(controller.face_contact),
    }


def _official_success(controller):
    site = controller.data.site_xpos[controller.site_id]
    stick = controller.data.xpos[controller.stick_id]
    arm_clear = float(np.linalg.norm(site[:2] - stick[:2])) > ARM_CLEAR_XY
    return bool(
        controller.succeeded
        and controller.phase == "done"
        and controller._cube_on_pad()
        and controller._stick_standing_on_cube()
        and not controller._stick_held()
        and not controller._hand_hits_stick()
        and arm_clear
    )


def _should_stop_no_cube(controller, data):
    cube = data.xpos[controller.cube_id]
    off_table = float(cube[0]) > 1.2 or float(np.linalg.norm(cube[:2] - np.array([0.5, 0.0]))) > 0.85
    if controller.phase == "done":
        return True
    if off_table and controller.phase in (
        "press",
        "push",
        "recover",
        "approach_cube",
        "lower",
    ):
        return bool(controller.lifted or data.time > 18.0)
    return data.time > 25.0


def run_episode(model, kind, rng, seconds, verbose, layout_hint=None):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    variant = randomize_episode(model, data, kind, rng, layout_hint=layout_hint)
    mujoco.mj_forward(model, data)
    init_state = data.qpos.copy()
    instruction, paraphrase_id = instruction_for(kind, rng)

    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink if not verbose else sys.stdout)
    with ctx:
        controller = Task3HandController(model, data)
    model.opt.timestep = CONTROL_DT

    use_expert = kind == "success" or kind in EXPERT_FAIL_KINDS
    policy = None if use_expert else BoundaryPolicy(controller, kind)

    act_ids, act_qadr = action_indices(model, controller.act_of_joint)
    rewards = RewardTracker(
        controller.table_top,
        controller.cube_half,
        controller.pad_half,
        controller.stick_half,
    )
    recorder = EpisodeRecorder(model, reward_terms=REWARD_TERMS, task_id=TASK_ID)

    steps = int(seconds / CONTROL_DT)
    for _ in range(steps):
        with ctx:
            if use_expert:
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
        if kind == "too_far":
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

        if use_expert:
            if controller.phase == "done" and controller.phase_time > 0.4:
                break
            if kind == "no_cube" and _should_stop_no_cube(controller, data):
                break
        elif policy.done:
            break

    if kind == "success":
        is_success = _official_success(controller)
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
        "action_mode": "delta_q" if use_expert else policy.action_mode,
        "fps": RECORD_HZ,
        "sim_time": float(data.time),
        "controller_phase": controller.phase,
        "placed": bool(controller._cube_on_pad()),
        "stick_planted": bool(controller._stick_standing_on_cube()),
        "succeeded_flag": bool(controller.succeeded),
        "init_state": init_state,
        **variant,
    }
    return recorder, meta, sink.getvalue()


def _validate_smoke(hdf5_path, model, rows):
    import h5py

    errors = []
    model_nq = int(model.nq)
    success_subs = []
    fail_kinds = []
    for row in rows:
        label = f"{row.get('group')}/{row.get('demo_id')}"
        if row["num_steps"] < 1:
            errors.append(f"{label} has no steps")
        if row["episode_kind"] == "success" and not row["is_success"]:
            errors.append(f"{label} success kind but is_success=false")
        if row["episode_kind"] != "success":
            if row["is_success"]:
                errors.append(f"{label} failure kind marked success")
            if not row["failure_reason"]:
                errors.append(f"{label} missing failure_reason")
            fail_kinds.append(row["episode_kind"])
        if row["episode_kind"] == "success" and row["is_success"]:
            if row["return"] <= 0.0:
                errors.append(f"{label} success return is not positive")
            success_subs.append(row.get("layout_subtype"))
            if row.get("layout_subtype") == "texture" and row.get("table_texture") != "checker":
                errors.append(f"{label} texture success did not use checker table")
            if row.get("layout_subtype") == "sized":
                if abs(float(row.get("stick_half", 0.10)) - 0.10) < 1e-4 and abs(
                    float(row.get("cube_half", 0.04)) - 0.04
                ) < 1e-4:
                    errors.append(f"{label} sized success still at nominal sizes")
            if row.get("layout_subtype") == "place":
                stick = np.asarray(row.get("stick_xy", [0.40, 0.26]), float)
                cube = np.asarray(row.get("cube_xy", [0.40, 0.10]), float)
                if (
                    float(np.linalg.norm(stick - np.array([0.40, 0.26]))) < 1e-4
                    and float(np.linalg.norm(cube - np.array([0.40, 0.10]))) < 1e-4
                ):
                    errors.append(f"{label} place success still at nominal XY")
    if success_subs.count(None) or set(success_subs) != {
        "standard",
        "lightness",
        "texture",
        "place",
        "sized",
    }:
        errors.append(f"success subtypes {success_subs} are not the required 5")
    if len(success_subs) != 5 or len(fail_kinds) != 5:
        errors.append(f"expected 5 success / 5 fail, got {len(success_subs)}/{len(fail_kinds)}")
    expect_fail = {
        "swapped",
        "too_far",
        "space_constraint",
        "cube_too_small",
        "stick_too_short",
    }
    if set(fail_kinds) != expect_fail:
        errors.append(f"fail kinds {fail_kinds} are not the required 5")

    with h5py.File(hdf5_path, "r") as handle:
        for gname, expect_ok in (("data", True), ("data_fail", False)):
            group = handle[gname]
            if "language_instruction" not in group.attrs:
                errors.append(f"{gname} missing language_instruction")
            if "language_instruction_subtitle" not in group.attrs:
                errors.append(f"{gname} missing language_instruction_subtitle")
            if str(group.attrs.get("task_id")) != TASK_ID:
                errors.append(f"{gname} task_id != task3")
            demos = sorted(
                [k for k in group.keys() if k.startswith("demo_")],
                key=lambda k: int(k.split("_")[1]),
            )
            for name in demos:
                demo = group[name]
                if demo["actions"].shape[1] != 15:
                    errors.append(f"{gname}/{name} actions dim != 15")
                if demo["obs/agentview_rgb"].shape[1:] != (224, 224, 3):
                    errors.append(f"{gname}/{name} bad agentview shape")
                if demo["states"].shape[1] != model_nq:
                    errors.append(f"{gname}/{name} states width != nq")
                if int(demo["dones"][-1]) != 1:
                    errors.append(f"{gname}/{name} dones[-1] != 1")
                ok = bool(demo.attrs["is_success"])
                if ok != expect_ok:
                    errors.append(f"{gname}/{name} is_success={ok} in {gname}")
                if int(demo["rewards"][-1]) != int(ok):
                    errors.append(f"{gname}/{name} rewards[-1] != is_success")
                if not demo_gripper_is_left(demo, model):
                    errors.append(f"{gname}/{name} gripper_states is not the left Dex3")
                if not demo.attrs.get("language_instruction"):
                    errors.append(f"{gname}/{name} missing language_instruction")
                if expect_ok and not demo.attrs.get("required_type"):
                    errors.append(f"{gname}/{name} missing required_type")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Collect Task 3 VLA episodes")
    parser.add_argument("--out", default="datasets/task3/smoke")
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
    hdf5_path = os.path.join(out_root, "plant_stick_on_cube_task3.hdf5")
    writer = LiberoPackWriter(
        hdf5_path,
        model.nq,
        task_id=TASK_ID,
        language_instruction=CANONICAL_INSTRUCTION,
        env_name="g1_stick_push",
    )

    print(
        f"collecting {args.n} Task 3 episode(s) to {hdf5_path} "
        f"(seed={args.seed}, {args.seconds:.0f}s cap)",
        flush=True,
    )
    for index in range(args.n):
        kind, layout_hint = _kind_for(index, args.n, kinds)
        model = mujoco.MjModel.from_xml_path(SCENE)
        recorder, meta, _log = run_episode(
            model,
            kind,
            rng,
            args.seconds,
            args.verbose,
            layout_hint=layout_hint,
        )
        if kind == "success" and not meta["is_success"]:
            for retry in range(3):
                model = mujoco.MjModel.from_xml_path(SCENE)
                hint = layout_hint
                if retry >= 1 and layout_hint:
                    hint = f"safe_{layout_hint}"
                recorder, meta, _log = run_episode(
                    model,
                    kind,
                    rng,
                    args.seconds,
                    args.verbose,
                    layout_hint=hint,
                )
                if meta["is_success"]:
                    break
        row = writer.append(recorder, meta)
        print(
            f"  {row['group']}/{row['demo_id']}  {kind:18s}  "
            f"{meta.get('layout_subtype', '?'):18s}  "
            f"{'PASS' if row['is_success'] else 'FAIL':4s}  "
            f"reason={row['failure_reason']}  "
            f"T={row['num_steps']:4d}  R={row['return']:+6.2f}  "
            f"pad={row['pad_xy']}  "
            f"stick={meta.get('stick_half', 0):.3f} cube={meta.get('cube_half', 0):.3f}",
            flush=True,
        )

    manifest_path = os.path.join(out_root, "manifest.json")
    writer.write_manifest(manifest_path)
    writer.finalize()
    errors = _validate_smoke(hdf5_path, model, writer.rows)
    print()
    print(f"wrote {len(writer.rows)} demos to {hdf5_path}")
    if errors:
        print("SMOKE CHECKS FAILED:")
        for err in errors:
            print(f"  {err}")
        return 1
    print("SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
