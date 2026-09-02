"""Closed-loop Task 1 eval: pretrained SmolVLA (7-D EE) or HDF5 replay.

SmolVLA path never calls Task1HandController.step(). Relative EE pose goes
through Jacobian IK; the gripper scalar interpolates the seven Dex3 joints.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python eval_smolvla_task1.py \\
        --policy replay --hdf5 datasets/task1/smoke/pick_place_task1.hdf5

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python eval_smolvla_task1.py \\
        --policy lerobot/smolvla_libero --n 1 --seconds 30 --seed 0
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

import mujoco
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

from task1_hand_controller import Task1HandController

from data_collection.randomize import (
    JUNK_NAMES,
    PARK_XY,
    TABLE_TOP,
    _set_free_pose,
    _set_pad_xy,
    _yaw_quat,
    randomize_episode,
)
from data_collection.recorder import (
    EpisodeRecorder,
    action_indices,
    fingers_closing,
    hand_hits_cube,
)
from data_collection.rewards import RewardTracker
from data_collection.schema import (
    ACTION_CLIP,
    CONTROL_DT,
    RECORD_HZ,
    RECORD_STRIDE,
    SUCCESS_INSTRUCTIONS,
)


SCENE = os.path.join(TASK_DIR, "task1_collect.xml")
CANONICAL_INSTRUCTION = SUCCESS_INSTRUCTIONS[0]
DEFAULT_HDF5 = os.path.join(
    TASK_DIR, "datasets/task1/smoke/pick_place_task1.hdf5"
)
DEFAULT_OUT = os.path.join(TASK_DIR, "datasets/task1/eval_smolvla")
VIDEO_SIZE = 480
VIDEO_FPS = float(RECORD_HZ)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _snapshot(controller, action, phase):
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
        "phase_time": float(controller.phase_time),
        "holding": bool(controller.holding),
        "grip_firm": bool(controller._grip_is_firm()),
        "fingers_closing": fingers_closing(controller),
        "hand_hits_cube": hand_hits_cube(controller),
        "cube_on_pad": bool(controller._cube_on_pad()),
        "cube_tipped": bool(controller._cube_tipped()),
        "action": np.asarray(action, dtype=np.float32),
    }


def _pack_obs(controller, recorder):
    scene = recorder._render(controller.data, recorder.cam_scene)
    ego = recorder._render(controller.data, recorder.cam_ego)
    pose = controller.ee_state()
    if pose.shape != (6,):
        raise RuntimeError(f"EE pose must be 6-D [x,y,z,rx,ry,rz], got {pose.shape}")
    # Checkpoint config says 6-D, but the loaded MEAN_STD stats are 8-D
    # (LIBERO: eef xyz + axis-angle + 2-finger gripper). Keep the first 6
    # as site pose and append two left Dex3 joints as the gripper pair.
    grip = []
    for name in ("left_hand_thumb_0_joint", "left_hand_index_0_joint"):
        act_id = controller.act_of_joint[name]
        grip.append(float(controller.data.qpos[controller.qadr[act_id]]))
    state = np.concatenate([pose, np.asarray(grip, dtype=np.float32)]).astype(np.float32)
    if state.shape != (8,):
        raise RuntimeError(f"observation.state must be 8-D for smolvla_libero stats, got {state.shape}")
    return scene, ego, state


def _cube_pad_xy(controller):
    cube = controller.data.xpos[controller.cube_id]
    pad = controller.data.xpos[controller.target_id]
    return float(np.linalg.norm(cube[:2] - pad[:2]))


def _draw_label(bgr, text, org, scale=0.45):
    import cv2

    x, y = org
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
        cv2.putText(
            bgr,
            text,
            (x + dx, y + dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA
    )


def _compose_video_frame(scene_rgb, ego_rgb, lines):
    import cv2

    left = np.asarray(scene_rgb, dtype=np.uint8)
    right = np.asarray(ego_rgb, dtype=np.uint8)
    pair = np.concatenate([left, right], axis=1)
    bgr = np.ascontiguousarray(pair[:, :, ::-1])
    _draw_label(bgr, "scene", (8, 22), 0.5)
    _draw_label(bgr, "wrist", (left.shape[1] + 8, 22), 0.5)
    y = 46
    for line in lines:
        _draw_label(bgr, line[:90], (8, y), 0.42)
        y += 20
    return bgr


def _render_video_pair(renderer, data, cam_scene, cam_ego):
    renderer.update_scene(data, camera=cam_scene)
    scene = renderer.render().copy()
    renderer.update_scene(data, camera=cam_ego)
    ego = renderer.render().copy()
    return scene, ego


def _write_mp4(path, frames, fps=VIDEO_FPS):
    if not frames:
        return None
    height, width = frames[0].shape[:2]
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        for code in ("mp4v", "avc1", "XVID"):
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*code), float(fps), (width, height)
            )
            if not writer.isOpened():
                continue
            for frame in frames:
                writer.write(frame)
            writer.release()
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return path
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    rgb = [frame[:, :, ::-1] for frame in frames]
    imageio.mimsave(path, rgb, fps=fps)
    return path


def _manifest_demo(hdf5_path, demo_id):
    path = os.path.join(os.path.dirname(hdf5_path), "manifest.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    for row in payload.get("demos") or []:
        if str(row.get("demo_id")) == str(demo_id) and row.get("group") == "data":
            return row
    return {}


def _restore_replay_layout(model, data, meta):
    pad_xy = meta.get("pad_xy")
    if pad_xy is not None and len(pad_xy) >= 2:
        _set_pad_xy(model, pad_xy)
    placed = {item["name"]: item for item in (meta.get("distractors") or []) if "name" in item}
    for i, name in enumerate(JUNK_NAMES):
        item = placed.get(name)
        if item is None:
            _set_free_pose(
                model,
                data,
                name,
                [PARK_XY[i][0], PARK_XY[i][1], 0.03],
                _yaw_quat(0.0),
            )
        else:
            xy = item["xy"]
            _set_free_pose(
                model,
                data,
                name,
                [float(xy[0]), float(xy[1]), TABLE_TOP + 0.03],
                _yaw_quat(0.0),
            )
    mujoco.mj_forward(model, data)


def _load_replay_demo(hdf5_path, demo_id):
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        group = handle["data"]
        if demo_id not in group:
            available = sorted(k for k in group.keys() if k.startswith("demo_"))
            raise KeyError(f"{demo_id} not in {hdf5_path}; have {available}")
        demo = group[demo_id]
        actions = np.asarray(demo["actions"], dtype=np.float32)
        joint = np.asarray(demo["obs/joint_states"], dtype=np.float32)
        init_state = demo.attrs.get("init_state")
        if init_state is not None:
            init_state = np.asarray(init_state, dtype=np.float64)
        instruction = str(
            demo.attrs.get("language_instruction") or CANONICAL_INSTRUCTION
        )
    meta = _manifest_demo(hdf5_path, demo_id)
    return actions, joint, init_state, instruction, meta


def _load_smolvla(model_id, device):
    import torch

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pretrained import PreTrainedConfig
    from lerobot.policies.smolvla import SmolVLAPolicy
    from lerobot.policies.utils import prepare_observation_for_inference

    config = PreTrainedConfig.from_pretrained(model_id)
    config.device = str(device)
    action_dim = int(config.action_feature.shape[0])
    if action_dim != 7:
        raise RuntimeError(
            f"{model_id} action dim is {action_dim}, expected 7 (do not retarget)"
        )
    state_dim = int(config.input_features["observation.state"].shape[0])
    if state_dim not in (6, 8):
        raise RuntimeError(
            f"{model_id} observation.state dim is {state_dim}, expected 6 or 8"
        )

    policy = SmolVLAPolicy.from_pretrained(model_id, config=config)
    policy.to(device)
    policy.eval()
    policy.reset()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocess, postprocess, prepare_observation_for_inference, torch


def _smolvla_action(
    policy,
    preprocess,
    postprocess,
    prepare_observation_for_inference,
    torch,
    scene,
    ego,
    state,
    instruction,
    device,
):
    raw = {
        "observation.state": np.asarray(state, dtype=np.float32),
        "observation.images.camera1": np.asarray(scene, dtype=np.uint8),
        "observation.images.camera2": np.asarray(ego, dtype=np.uint8),
    }
    frame = prepare_observation_for_inference(
        raw, device=device, task=instruction, robot_type="g1"
    )
    batch = preprocess(frame)
    with torch.inference_mode():
        action = policy.select_action(batch)
    action = postprocess(action)
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size < 7:
        raise RuntimeError(f"policy returned {action.size}-D action, expected 7")
    return action[:7]


def run_episode(
    model,
    *,
    seconds,
    verbose,
    policy_kind,
    rng,
    replay_actions=None,
    replay_qpos=None,
    init_state=None,
    replay_meta=None,
    instruction=CANONICAL_INSTRUCTION,
    smol=None,
    device=None,
    save_video=False,
):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if init_state is not None:
        if replay_meta:
            _restore_replay_layout(model, data, replay_meta)
        data.qpos[:] = np.asarray(init_state, dtype=np.float64).reshape(-1)[: model.nq]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        variant = {"layout_subtype": "replay"}
    else:
        variant = randomize_episode(model, data, "success", rng)
        mujoco.mj_forward(model, data)

    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink if not verbose else sys.stdout)
    expert_init = policy_kind == "replay"
    with ctx:
        controller = Task1HandController(model, data, expert_init=expert_init)
    if expert_init is False and controller.expert_init:
        raise RuntimeError("SmolVLA path must construct the controller with expert_init=False")
    controller.phase = "policy"
    model.opt.timestep = CONTROL_DT

    act_ids, act_qadr = action_indices(model, controller.act_of_joint)
    rewards = RewardTracker(
        controller.table_top, controller.cube_half, controller.pad_half
    )
    recorder = EpisodeRecorder(model)
    video_renderer = None
    video_frames = []
    if save_video:
        video_renderer = mujoco.Renderer(model, VIDEO_SIZE, VIDEO_SIZE)

    steps = int(seconds / CONTROL_DT)
    action = np.zeros(7 if policy_kind == "smolvla" else 15, dtype=np.float32)
    action_log = []
    grasped = False
    started = time.perf_counter()
    replay_i = 0

    for tick in range(steps):
        policy_tick = tick % RECORD_STRIDE == 0
        if policy_tick:
            if policy_kind == "replay":
                if replay_actions is None or replay_i >= len(replay_actions):
                    break
                action = np.asarray(replay_actions[replay_i], dtype=np.float32)
                if replay_qpos is not None and replay_i < len(replay_qpos):
                    ref = np.asarray(replay_qpos[replay_i], dtype=np.float32)[act_ids]
                else:
                    ref = None
                replay_i += 1
                controller.apply_delta_q(
                    action, act_ids, act_qadr, ACTION_CLIP, ref_qpos=ref
                )
            else:
                scene, ego, state = _pack_obs(controller, recorder)
                action = _smolvla_action(
                    smol["policy"],
                    smol["preprocess"],
                    smol["postprocess"],
                    smol["prepare"],
                    smol["torch"],
                    scene,
                    ego,
                    state,
                    instruction,
                    device,
                )
                controller.set_libero_action(action)
            action_log.append(np.asarray(action, dtype=np.float32).copy())

        with ctx:
            controller.pd_tick(slew=(policy_kind == "smolvla"))
        mujoco.mj_step(model, data)
        controller.refresh_hold_flag()
        if controller.holding:
            grasped = True

        if policy_tick:
            snap = _snapshot(controller, action, controller.phase)
            reward, terms, events = rewards.step(snap, "success")
            recorder.maybe_record(
                data, controller, action, controller.phase, reward, terms, events
            )
            if video_renderer is not None:
                scene_v, ego_v = _render_video_pair(
                    video_renderer,
                    data,
                    recorder.cam_scene,
                    recorder.cam_ego,
                )
                video_frames.append(
                    _compose_video_frame(
                        scene_v,
                        ego_v,
                        [
                            f"{policy_kind}  t={data.time:.1f}s",
                            instruction,
                            (
                                f"holding={int(controller.holding)}  "
                                f"grasped={int(grasped)}  "
                                f"placed={int(controller._cube_on_pad())}  "
                                f"xy={_cube_pad_xy(controller):.3f}m"
                            ),
                        ],
                    )
                )
        else:
            recorder.tick += 1

        if controller._cube_on_pad() and controller.holding is False:
            # placed and released, or sitting still on the pad
            if controller._cube_speed() < 0.05:
                break

    placed = bool(controller._cube_on_pad())
    cube = data.xpos[controller.cube_id].copy()
    pad = data.xpos[controller.target_id].copy()
    stacked = (
        np.stack(action_log, axis=0)
        if action_log
        else np.zeros((0, action.size), dtype=np.float32)
    )
    fired = sorted(rewards.fired)
    result = {
        "policy": policy_kind,
        "placed": placed,
        "grasped": bool(grasped),
        "holding": bool(controller.holding),
        "cube_pad_xy": _cube_pad_xy(controller),
        "cube": cube,
        "pad": pad,
        "return": float(np.sum(recorder.rewards)) if recorder.num_steps else 0.0,
        "fired_terms": fired,
        "mean_abs_action": float(np.mean(np.abs(stacked))) if stacked.size else 0.0,
        "mean_gripper": (
            float(np.mean(stacked[:, -1]))
            if stacked.size and stacked.shape[1] >= 7
            else None
        ),
        "num_policy_steps": int(len(action_log)),
        "sim_time": float(data.time),
        "wall_time": float(time.perf_counter() - started),
        "instruction": instruction,
        "expert_init": bool(controller.expert_init),
        "observation_state_dim": 8 if policy_kind == "smolvla" else 6,
        "action_dim": int(stacked.shape[1]) if stacked.size else int(action.size),
        **{k: v for k, v in variant.items() if not isinstance(v, np.ndarray)},
    }
    if video_renderer is not None:
        video_renderer.close()
    return result, recorder, sink.getvalue(), video_frames


def main():
    parser = argparse.ArgumentParser(description="Task 1 SmolVLA / replay eval")
    parser.add_argument(
        "--policy",
        default="lerobot/smolvla_libero",
        help="lerobot/smolvla_libero or 'replay'",
    )
    parser.add_argument("--hdf5", default=DEFAULT_HDF5)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a 10 fps side-by-side cam_scene/cam_ego MP4 (default: on)",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    policy_kind = "replay" if args.policy.strip().lower() == "replay" else "smolvla"

    smol = None
    device = None
    replay_actions = None
    replay_qpos = None
    replay_meta = None
    init_state = None
    instruction = CANONICAL_INSTRUCTION

    if policy_kind == "replay":
        replay_actions, replay_qpos, init_state, instruction, replay_meta = (
            _load_replay_demo(args.hdf5, args.demo)
        )
        n_ep = 1
        seconds = min(args.seconds, (len(replay_actions) + 2) * RECORD_STRIDE * CONTROL_DT)
        print(
            f"replay {args.demo} from {args.hdf5} "
            f"({len(replay_actions)} actions, {seconds:.1f}s cap)",
            flush=True,
        )
    else:
        n_ep = args.n
        seconds = args.seconds
        print(f"loading {args.policy} on {args.device} ...", flush=True)
        policy, preprocess, postprocess, prepare, torch = _load_smolvla(
            args.policy, args.device
        )
        import torch as torch_mod

        device = torch_mod.device(args.device)
        smol = {
            "policy": policy,
            "preprocess": preprocess,
            "postprocess": postprocess,
            "prepare": prepare,
            "torch": torch_mod,
        }
        print(
            f"running {n_ep} SmolVLA episode(s), {seconds:.0f}s cap, "
            f"instruction={instruction!r}",
            flush=True,
        )

    rows = []
    for index in range(n_ep):
        model = mujoco.MjModel.from_xml_path(SCENE)
        if policy_kind == "smolvla" and smol is not None:
            smol["policy"].reset()
        result, _recorder, _log, video_frames = run_episode(
            model,
            seconds=seconds,
            verbose=args.verbose,
            policy_kind=policy_kind,
            rng=rng,
            replay_actions=replay_actions,
            replay_qpos=replay_qpos,
            init_state=init_state,
            replay_meta=replay_meta,
            instruction=instruction,
            smol=smol,
            device=device,
            save_video=args.save_video,
        )
        result["episode_index"] = index
        if args.save_video:
            video_name = f"{policy_kind}_seed{args.seed}.mp4"
            if n_ep > 1:
                video_name = f"{policy_kind}_seed{args.seed}_ep{index}.mp4"
            video_path = os.path.join(args.out, video_name)
            written = _write_mp4(video_path, video_frames, VIDEO_FPS)
            result["video"] = written
            if written:
                print(f"  wrote {written} ({len(video_frames)} frames)", flush=True)
            else:
                print("  video skipped (no frames)", flush=True)
        rows.append(result)
        print(
            f"  ep{index}  {'SUCCESS' if result['placed'] else 'FAILURE':7s}  "
            f"placed={result['placed']} grasped={result['grasped']}  "
            f"xy={result['cube_pad_xy']:.4f}  R={result['return']:+6.2f}  "
            f"steps={result['num_policy_steps']}  t={result['sim_time']:.1f}s  "
            f"wall={result['wall_time']:.1f}s  fired={result['fired_terms']}",
            flush=True,
        )

    n_ok = sum(1 for row in rows if row["placed"])
    summary = {
        "policy": args.policy,
        "policy_kind": policy_kind,
        "n": len(rows),
        "n_success": n_ok,
        "success_rate": float(n_ok) / max(len(rows), 1),
        "instruction": instruction,
        "observation_state": "[x, y, z, rx, ry, rz, grip0, grip1]",
        "observation_state_dim": 8,
        "smolvla_action_dim": 7 if policy_kind == "smolvla" else 15,
        "episodes": rows,
    }
    out_path = os.path.join(args.out, f"{policy_kind}_seed{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2)
    print()
    print(f"wrote {out_path}")
    print(
        f"RESULT: {n_ok}/{len(rows)} placed "
        f"(rate={summary['success_rate']:.2f})"
    )
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
