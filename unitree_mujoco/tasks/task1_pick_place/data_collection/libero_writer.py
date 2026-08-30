"""Pack Task 1 episodes into one robomimic/LIBERO-style HDF5."""

import json
import os

import h5py
import numpy as np

from .schema import ACTION_JOINTS, IMAGE_SIZE, RECORD_HZ, SUCCESS_INSTRUCTIONS, TASK_ID


CANONICAL_INSTRUCTION = SUCCESS_INSTRUCTIONS[0]
SUBTITLE_SUCCESS = "success demonstrations"
SUBTITLE_FAIL = "boundary failures (do not use for BC)"
GRIPPER_JOINTS = ACTION_JOINTS[-7:]


def _event_mask(recorder):
    mask = np.zeros((len(recorder.events), len(recorder.term_names)), dtype=np.uint8)
    name_to_i = {n: i for i, n in enumerate(recorder.term_names)}
    for t, evs in enumerate(recorder.events):
        for name in evs:
            if name in name_to_i:
                mask[t, name_to_i[name]] = 1
    return mask


def _phase_returns(recorder):
    rewards = np.asarray(recorder.rewards, dtype=np.float32)
    phases = np.asarray(recorder.phases)
    out = {}
    for phase in sorted(set(recorder.phases)):
        mask = phases == phase
        out[phase] = float(rewards[mask].sum()) if mask.any() else 0.0
    return out


class LiberoPackWriter:
    """Write successes under data/ and failures under data_fail/."""

    def __init__(
        self,
        path,
        nq,
        task_id=TASK_ID,
        language_instruction=CANONICAL_INSTRUCTION,
        env_name="g1_pick_place",
    ):
        self.path = path
        self.nq = int(nq)
        self.task_id = task_id
        self.language_instruction = language_instruction
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.handle = h5py.File(path, "w")
        self.data = self.handle.create_group("data")
        self.data_fail = self.handle.create_group("data_fail")
        self._n_ok = 0
        self._n_fail = 0
        self._steps_ok = 0
        self._steps_fail = 0
        self.rows = []
        self._stamp_group(self.data, SUBTITLE_SUCCESS, env_name)
        self._stamp_group(self.data_fail, SUBTITLE_FAIL, env_name)

    def _stamp_group(self, group, subtitle, env_name):
        group.attrs["language_instruction"] = self.language_instruction
        group.attrs["language_instruction_subtitle"] = subtitle
        group.attrs["task_id"] = self.task_id
        group.attrs["env_name"] = env_name
        group.attrs["fps"] = float(RECORD_HZ)
        group.attrs["action_space"] = "delta_q"
        group.attrs["action_joints"] = [n.encode() for n in ACTION_JOINTS]
        group.attrs["image_size"] = int(IMAGE_SIZE)

    def append(self, recorder, meta):
        is_success = bool(meta.get("is_success"))
        group = self.data if is_success else self.data_fail
        index = self._n_ok if is_success else self._n_fail
        name = f"demo_{index}"
        demo = group.create_group(name)

        t = recorder.num_steps
        if t < 1:
            raise ValueError("cannot append an empty episode")

        actions = np.asarray(recorder.actions, np.float32)
        joint = np.asarray(recorder.qpos, np.float32)
        ee_pos = np.asarray(recorder.ee_pos, np.float32)
        ee_ori = np.asarray(recorder.ee_ori, np.float32)
        if ee_ori.size == 0:
            ee_ori = np.zeros((t, 3), dtype=np.float32)
        states = np.asarray(recorder.full_qpos, np.float32)
        if states.size == 0:
            states = np.zeros((t, self.nq), dtype=np.float32)
        gripper = joint[:, -len(GRIPPER_JOINTS) :]
        ee_states = np.concatenate([ee_pos, ee_ori], axis=1)
        robot_states = np.concatenate([gripper, ee_pos, ee_ori], axis=1)

        dones = np.zeros((t,), dtype=np.uint8)
        dones[-1] = 1
        rewards = np.zeros((t,), dtype=np.uint8)
        if is_success:
            rewards[-1] = 1
        dense = np.asarray(recorder.rewards, np.float32)
        terms = np.asarray(recorder.reward_terms, np.float32)

        def _ds(key, array, **kwargs):
            demo.create_dataset(key, data=array, **kwargs)

        _ds(
            "obs/agentview_rgb",
            np.asarray(recorder.images_scene, np.uint8),
            compression="gzip",
            compression_opts=1,
        )
        _ds(
            "obs/eye_in_hand_rgb",
            np.asarray(recorder.images_ego, np.uint8),
            compression="gzip",
            compression_opts=1,
        )
        _ds("obs/joint_states", joint)
        _ds("obs/gripper_states", gripper)
        _ds("obs/ee_pos", ee_pos)
        _ds("obs/ee_ori", ee_ori)
        _ds("obs/ee_states", ee_states)
        _ds("actions", actions)
        _ds("dones", dones)
        _ds("rewards", rewards)
        demo["dones"][-1] = 1
        if is_success:
            demo["rewards"][-1] = 1
        _ds("states", states)
        _ds("robot_states", robot_states)
        _ds("phase", np.asarray(recorder.phases, dtype="S24"))
        _ds("reward_dense", dense)
        _ds("reward_terms", terms)
        _ds("events", _event_mask(recorder))

        instruction = str(meta.get("instruction") or self.language_instruction)
        demo.attrs["num_samples"] = t
        demo.attrs["is_success"] = is_success
        demo.attrs["failure_reason"] = str(meta.get("failure_reason") or "")
        demo.attrs["episode_kind"] = str(meta.get("episode_kind") or "")
        demo.attrs["instruction"] = instruction
        demo.attrs["language_instruction"] = instruction
        if meta.get("required_type"):
            demo.attrs["required_type"] = str(meta["required_type"])
        if meta.get("chosen_type"):
            demo.attrs["chosen_type"] = str(meta["chosen_type"])
        init_state = meta.get("init_state")
        if init_state is not None:
            demo.attrs["init_state"] = np.asarray(init_state, np.float32)
        demo.attrs["reward_term_names"] = [n.encode() for n in recorder.term_names]
        demo.attrs["action_joints"] = [n.encode() for n in ACTION_JOINTS]

        if is_success:
            self._n_ok += 1
            self._steps_ok += t
        else:
            self._n_fail += 1
            self._steps_fail += t

        row = {
            "group": "data" if is_success else "data_fail",
            "demo_id": name,
            "num_steps": t,
            "return": float(dense.sum()),
            "phase_returns": _phase_returns(recorder),
            **{
                k: v
                for k, v in meta.items()
                if k != "init_state" and not isinstance(v, np.ndarray)
            },
        }
        self.rows.append(row)
        self.handle.flush()
        return row

    def finalize(self):
        self.data.attrs["num_demos"] = self._n_ok
        self.data.attrs["total"] = self._steps_ok
        self.data_fail.attrs["num_demos"] = self._n_fail
        self.data_fail.attrs["total"] = self._steps_fail
        self.handle.close()

    def write_manifest(self, path):
        payload = {
            "task_id": self.task_id,
            "language_instruction": self.language_instruction,
            "hdf5": os.path.basename(self.path),
            "num_success": self._n_ok,
            "num_fail": self._n_fail,
            "demos": self.rows,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return payload
