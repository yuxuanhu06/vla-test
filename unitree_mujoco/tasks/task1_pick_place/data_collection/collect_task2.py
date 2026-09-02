"""Collect Task 2 VLA episodes (success + flagged boundary failures).

    python collect_task2.py --out datasets/task2/smoke --n 10 --seed 0
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import random
import sys

import mujoco
import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

from task2_hand_controller import Task2HandController

from data_collection.boundary_policy_task2 import BoundaryPolicy
from data_collection.libero_writer import LiberoPackWriter, demo_gripper_is_left
from data_collection.randomize_task2 import BODY_OF, randomize_episode
from data_collection.recorder import (
    EpisodeRecorder,
    action_indices,
    compute_action,
    fingers_closing,
    hand_hits_cube,
)
from data_collection.rewards_task2 import RewardTracker
from data_collection.schema_task2 import (
    CANONICAL_INSTRUCTION,
    CONTROL_DT,
    OBJECT_TYPES,
    RECORD_HZ,
    REWARD_TERMS,
    SMOKE_SPECS,
    TASK_ID,
    instruction_for,
)


SCENE = os.path.join(TASK_DIR, "task2_collect.xml")
NEIGHBOR_MOVE_TOL = 0.02


def _kind_for(index, n, kinds):
    if n == len(SMOKE_SPECS) and not kinds:
        return SMOKE_SPECS[index][0], SMOKE_SPECS[index][1], SMOKE_SPECS[index][2]
    if kinds:
        return kinds[index % len(kinds)], None, None
    success_n = max(1, int(round(0.35 * n)))
    space_n = max(1, int(round(0.15 * n)))
    kin_n = max(1, int(round(0.15 * n)))
    tip_n = max(1, int(round(0.10 * n)))
    wrong_n = max(1, int(round(0.15 * n)))
    bounds = [
        (success_n, "success"),
        (space_n, "space_constraint"),
        (kin_n, "kinematics_limit"),
        (tip_n, "tipped"),
        (wrong_n, "wrong_object"),
    ]
    cursor = 0
    for count, kind in bounds:
        if index < cursor + count:
            hint = OBJECT_TYPES[index % 4] if kind == "success" else None
            return kind, None, hint
        cursor += count
    return "neighbor_collision", "too_close", None


def _object_xy(model, data, typ):
    return data.xpos[model.body(BODY_OF[typ]).id][:2].copy()


def _object_pos(model, data, typ):
    return data.xpos[model.body(BODY_OF[typ]).id].copy()


def _object_up(model, data, typ):
    body_id = model.body(BODY_OF[typ]).id
    return float(data.xmat[body_id].reshape(3, 3)[2, 2])


def _object_vel(model, data, typ):
    body_id = model.body(BODY_OF[typ]).id
    jnt = model.body_jntadr[body_id]
    vadr = model.jnt_dofadr[jnt]
    return data.qvel[vadr : vadr + 3].copy()


def _neighbors_ok(start_xy, model, data, skip_type, fallen_types):
    deltas = {}
    ok = True
    for typ in OBJECT_TYPES:
        if typ == skip_type:
            continue
        end = _object_xy(model, data, typ)
        delta = float(np.linalg.norm(end - start_xy[typ]))
        deltas[typ] = delta
        if delta > NEIGHBOR_MOVE_TOL:
            ok = False
        if typ not in fallen_types and _object_up(model, data, typ) < 0.8:
            ok = False
    return ok, deltas


def _hand_hits_other(controller, skip_geom):
    if skip_geom is None:
        return False
    hand = controller.thumb_geoms | controller.finger_geoms
    others = set()
    for _typ, (_body, geom_name) in Task2HandController.OBJECT_TYPES.items():
        gid = int(controller.model.geom(geom_name).id)
        if gid != skip_geom:
            others.add(gid)
    for i in range(controller.data.ncon):
        pair = {
            int(controller.data.contact[i].geom1),
            int(controller.data.contact[i].geom2),
        }
        if pair & hand and pair & others:
            return True
    return False


def _other_on_pad(controller, model, data, skip_type):
    pad = data.xpos[controller.target_id][:2]
    for typ in OBJECT_TYPES:
        if typ == skip_type:
            continue
        xy = _object_xy(model, data, typ)
        if float(np.linalg.norm(xy - pad)) < controller.pad_half - 0.02:
            return True
    return False


def _snapshot(controller, model, data, action, phase, phase_time, required_type, start_xy, fallen_types):
    chosen = required_type if controller.cube_id is None else None
    if controller.cube_id is not None:
        for typ, (body, _g) in Task2HandController.OBJECT_TYPES.items():
            if model.body(body).id == controller.cube_id:
                chosen = typ
                break
    cube = _object_pos(model, data, chosen or required_type)
    pad = data.xpos[controller.target_id].copy()
    site = data.site_xpos[controller.site_id].copy()
    on_pad = False
    tipped = False
    if controller.cube_id is not None:
        on_pad = bool(controller._cube_on_pad())
        tipped = bool(controller._cube_tipped())
    else:
        tipped = _object_up(model, data, required_type) < 0.8

    neighbor_ok = True
    if start_xy is not None:
        neighbor_ok, _ = _neighbors_ok(start_xy, model, data, required_type, fallen_types)

    return {
        "ee_pos": site,
        "cube_pos": cube,
        "pad_pos": pad,
        "cube_vel": _object_vel(model, data, chosen or required_type),
        "phase": phase,
        "phase_time": float(phase_time),
        "holding": bool(controller.holding),
        "grip_firm": bool(controller._grip_is_firm()) if controller.cube_id is not None else False,
        "fingers_closing": fingers_closing(controller),
        "hand_hits_cube": hand_hits_cube(controller),
        "cube_on_pad": on_pad,
        "cube_tipped": tipped,
        "action": action,
        "decision_ok": chosen == required_type if chosen is not None else False,
        "required_bound": controller.cube_id is not None,
        "holding_wrong": _hand_hits_other(controller, controller.cube_geom_id)
        and bool(controller.holding),
        "neighbors_disturbed": not neighbor_ok,
        "other_on_pad": _other_on_pad(controller, model, data, required_type),
    }


def run_episode(model, kind, rng, seconds, verbose, layout_hint=None, type_hint=None):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    variant = randomize_episode(
        model, data, kind, rng, layout_hint=layout_hint, type_hint=type_hint
    )
    mujoco.mj_forward(model, data)
    init_state = data.qpos.copy()
    required_type = variant["required_type"]
    instruction, paraphrase_id = instruction_for(kind, required_type, rng)

    original_choice = random.choice

    def _forced_choice(seq):
        items = list(seq)
        if required_type in items:
            return required_type
        return original_choice(items)

    random.choice = _forced_choice
    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink if not verbose else sys.stdout)
    try:
        with ctx:
            controller = Task2HandController(model, data)
        model.opt.timestep = CONTROL_DT

        use_expert = kind in ("success", "neighbor_collision")
        policy = None if use_expert else BoundaryPolicy(controller, kind)

        act_ids, act_qadr = action_indices(model, controller.act_of_joint)
        rewards = RewardTracker(
            controller.table_top, controller.cube_half, controller.pad_half
        )
        recorder = EpisodeRecorder(model, reward_terms=REWARD_TERMS, task_id=TASK_ID)

        start_xy = {typ: _object_xy(model, data, typ) for typ in OBJECT_TYPES}
        fallen_types = set(variant["fallen_types"])
        steps = int(seconds / CONTROL_DT)
        for _ in range(steps):
            with ctx:
                if use_expert:
                    controller.step()
                    phase = controller.phase
                    phase_time = controller.phase_time
                elif kind == "wrong_object":
                    # Do not run identify: the expert would bind the parked body.
                    policy.step(CONTROL_DT)
                    phase = policy.phase
                    phase_time = policy.phase_time
                    controller.phase = phase
                else:
                    if controller.cube_id is None:
                        controller.step()
                    else:
                        policy.step(CONTROL_DT)
                    phase = policy.phase if controller.cube_id is not None else controller.phase
                    phase_time = (
                        policy.phase_time
                        if controller.cube_id is not None
                        else controller.phase_time
                    )
                    controller.phase = phase
            mujoco.mj_step(model, data)

            action = compute_action(controller, act_ids, act_qadr)
            if kind == "kinematics_limit":
                action = np.zeros_like(action)
                controller.q_target[act_ids] = controller.data.qpos[act_qadr]

            if recorder.should_record():
                snap = _snapshot(
                    controller,
                    model,
                    data,
                    action,
                    phase,
                    phase_time,
                    required_type,
                    start_xy,
                    fallen_types,
                )
                reward, terms, events = rewards.step(snap, kind)
                recorder.maybe_record(
                    data, controller, action, phase, reward, terms, events
                )
            else:
                recorder.tick += 1

            if use_expert:
                if controller.phase == "done" and controller.phase_time > 0.4:
                    break
            elif policy.done:
                break
    finally:
        random.choice = original_choice

    chosen_type = None
    if controller.cube_id is not None:
        for typ, (body, _g) in Task2HandController.OBJECT_TYPES.items():
            if model.body(body).id == controller.cube_id:
                chosen_type = typ
                break

    placed = bool(controller.cube_id is not None and controller._cube_on_pad())
    neighbors_ok, neighbor_deltas = _neighbors_ok(
        start_xy, model, data, required_type, fallen_types
    )
    decision_ok = chosen_type == required_type

    if kind == "success":
        is_success = bool(
            decision_ok
            and placed
            and controller.phase == "done"
            and neighbors_ok
        )
        failure_reason = None if is_success else "space_constraint"
    elif kind == "neighbor_collision":
        is_success = False
        failure_reason = "knock"
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
        "placed": placed,
        "decision_ok": bool(decision_ok),
        "chosen_type": chosen_type,
        "neighbors_ok": bool(neighbors_ok),
        "neighbor_deltas": {k: float(v) for k, v in neighbor_deltas.items()},
        "init_state": init_state,
        **variant,
    }
    return recorder, meta, sink.getvalue()


def _validate_smoke(hdf5_path, model, rows):
    import h5py

    errors = []
    model_nq = int(model.nq)
    success_types = []
    success_pads = []
    has_mixed_far = False
    has_tipped_stand = False
    has_tipped_fail = False
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
        if row["episode_kind"] == "success" and row["is_success"]:
            if row["return"] <= 0.0:
                errors.append(f"{label} success return is not positive")
            success_types.append(row["required_type"])
            success_pads.append(tuple(np.round(row["pad_xy"], 4)))
            if row.get("layout_subtype") == "mixed_tipped":
                has_tipped_stand = True
        if row.get("layout_subtype") == "mixed_far":
            has_mixed_far = True
        if row["episode_kind"] == "tipped":
            has_tipped_fail = True
        if row["episode_kind"] == "wrong_object":
            missing = row.get("missing_types") or []
            if row["required_type"] not in missing:
                errors.append(f"{label} wrong_object but required type is still present")
            objects = row.get("objects") or {}
            on_table = [
                t
                for t, rec in objects.items()
                if rec.get("present", True)
            ]
            if row["required_type"] in on_table:
                errors.append(f"{label} requested {row['required_type']} is still on the table")
            if len(on_table) != 3:
                errors.append(f"{label} expected 3 objects on the table, got {on_table}")
    if len(success_types) >= 4 and set(success_types[:4]) != set(OBJECT_TYPES):
        errors.append(f"first success types not one of each: {success_types[:4]}")
    if success_pads and len(set(success_pads)) < 2:
        errors.append("all success pad_xy are identical")
    if not has_mixed_far:
        errors.append("smoke has no mixed_far episode")
    if not has_tipped_stand:
        errors.append("smoke has no mixed_tipped standing success")
    if not has_tipped_fail:
        errors.append("smoke has no tipped fallen fail")

    with h5py.File(hdf5_path, "r") as handle:
        for gname, expect_ok in (("data", True), ("data_fail", False)):
            group = handle[gname]
            if "language_instruction" not in group.attrs:
                errors.append(f"{gname} missing language_instruction")
            if "language_instruction_subtitle" not in group.attrs:
                errors.append(f"{gname} missing language_instruction_subtitle")
            if str(group.attrs.get("task_id")) != TASK_ID:
                errors.append(f"{gname} task_id != task2")
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
    parser = argparse.ArgumentParser(description="Collect Task 2 VLA episodes")
    parser.add_argument("--out", default="datasets/task2/smoke")
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
    hdf5_path = os.path.join(out_root, "pick_correct_object_task2.hdf5")
    writer = LiberoPackWriter(
        hdf5_path,
        model.nq,
        task_id=TASK_ID,
        language_instruction=CANONICAL_INSTRUCTION,
        env_name="g1_pick_by_type",
    )

    print(
        f"collecting {args.n} Task 2 episode(s) to {hdf5_path} "
        f"(seed={args.seed}, {args.seconds:.0f}s cap)",
        flush=True,
    )
    for index in range(args.n):
        kind, layout_hint, type_hint = _kind_for(index, args.n, kinds)
        model = mujoco.MjModel.from_xml_path(SCENE)
        recorder, meta, _log = run_episode(
            model,
            kind,
            rng,
            args.seconds,
            args.verbose,
            layout_hint=layout_hint,
            type_hint=type_hint,
        )
        if kind == "success" and not meta["is_success"]:
            for retry in range(2):
                model = mujoco.MjModel.from_xml_path(SCENE)
                hint = layout_hint
                if retry == 1 and layout_hint not in (
                    None,
                    "working_row",
                    "mixed_tipped",
                    "mixed_far",
                ):
                    hint = "working_row"
                recorder, meta, _log = run_episode(
                    model,
                    kind,
                    rng,
                    args.seconds,
                    args.verbose,
                    layout_hint=hint,
                    type_hint=type_hint,
                )
                if meta["is_success"]:
                    break
        row = writer.append(recorder, meta)
        print(
            f"  {row['group']}/{row['demo_id']}  {kind:18s}  "
            f"{meta.get('required_type', '?'):11s}  "
            f"{meta.get('layout_subtype', '?'):18s}  "
            f"{'PASS' if row['is_success'] else 'FAIL':4s}  "
            f"reason={row['failure_reason']}  "
            f"T={row['num_steps']:4d}  R={row['return']:+6.2f}  "
            f"pad={row['pad_xy']}",
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
