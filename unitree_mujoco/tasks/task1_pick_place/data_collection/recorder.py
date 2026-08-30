"""Offscreen cameras, 10 Hz buffering, and HDF5/JSON episode writes."""

import json
import os

import h5py
import mujoco
import numpy as np

from .schema import (
    ACTION_CLIP,
    ACTION_JOINTS,
    IMAGE_SIZE,
    RECORD_STRIDE,
    REWARD_TERMS,
    TASK_ID,
)


def action_indices(model, act_of_joint):
    ids = []
    qadr = []
    for name in ACTION_JOINTS:
        act_id = act_of_joint[name]
        ids.append(act_id)
        joint_id = model.actuator_trnid[act_id, 0]
        qadr.append(model.jnt_qposadr[joint_id])
    return np.array(ids, dtype=int), np.array(qadr, dtype=int)


def compute_action(controller, act_ids, act_qadr):
    q = controller.data.qpos[act_qadr]
    target = controller.q_target[act_ids]
    return np.clip(target - q, -ACTION_CLIP, ACTION_CLIP).astype(np.float32)


def hand_hits_cube(controller):
    cube = controller.cube_geom_id
    if cube is None:
        return False
    hand = controller.thumb_geoms | controller.finger_geoms
    for i in range(controller.data.ncon):
        pair = {
            int(controller.data.contact[i].geom1),
            int(controller.data.contact[i].geom2),
        }
        if cube in pair and pair & hand:
            return True
    return False


def mat_to_axis_angle(mat):
    """Convert a 3x3 rotation matrix to an axis-angle vector."""
    rot = np.asarray(mat, dtype=float).reshape(3, 3)
    cos = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=float,
    )
    axis = axis / (2.0 * np.sin(angle) + 1e-12)
    return (axis * angle).astype(np.float32)


def fingers_closing(controller):
    err = 0.0
    for act_id, closed in controller.finger_close_target.items():
        q = float(controller.data.qpos[controller.qadr[act_id]])
        opened = controller.finger_open_target[act_id]
        span = abs(closed - opened) + 1e-6
        err = max(err, abs(q - opened) / span)
    return err > 0.35


class EpisodeRecorder:
    def __init__(self, model, reward_terms=None, task_id=None):
        self.model = model
        self.term_names = tuple(reward_terms) if reward_terms is not None else REWARD_TERMS
        self.task_id = task_id if task_id is not None else TASK_ID
        self.renderer = mujoco.Renderer(model, IMAGE_SIZE, IMAGE_SIZE)
        self.cam_scene = model.camera("cam_scene").id
        self.cam_ego = model.camera("cam_ego").id
        self.reset()

    def reset(self):
        self.images_scene = []
        self.images_ego = []
        self.qpos = []
        self.qvel = []
        self.ee_pos = []
        self.ee_ori = []
        self.full_qpos = []
        self.actions = []
        self.phases = []
        self.rewards = []
        self.reward_terms = []
        self.events = []
        self.tick = 0

    def _render(self, data, camera_id):
        self.renderer.update_scene(data, camera=camera_id)
        return self.renderer.render().copy()

    def should_record(self):
        return self.tick % RECORD_STRIDE == 0

    def maybe_record(self, data, controller, action, phase, reward, terms, events):
        record = self.should_record()
        self.tick += 1
        if not record:
            return False
        self.images_scene.append(self._render(data, self.cam_scene))
        self.images_ego.append(self._render(data, self.cam_ego))
        self.qpos.append(data.qpos[controller.qadr].copy())
        self.qvel.append(data.qvel[controller.vadr].copy())
        self.ee_pos.append(data.site_xpos[controller.site_id].copy())
        self.ee_ori.append(
            mat_to_axis_angle(data.site_xmat[controller.site_id])
        )
        self.full_qpos.append(data.qpos.copy())
        self.actions.append(np.asarray(action, np.float32))
        self.phases.append(phase)
        self.rewards.append(float(reward))
        self.reward_terms.append([float(terms[name]) for name in self.term_names])
        self.events.append(list(events))
        return True

    @property
    def num_steps(self):
        return len(self.rewards)

    def write(self, out_dir, episode_id, meta):
        os.makedirs(out_dir, exist_ok=True)
        hdf5_path = os.path.join(out_dir, "episode.hdf5")
        json_path = os.path.join(out_dir, "episode.json")

        terms = np.asarray(self.reward_terms, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        phases = np.asarray(self.phases)
        phase_returns = {}
        for phase in sorted(set(self.phases)):
            mask = phases == phase
            phase_returns[phase] = float(rewards[mask].sum()) if mask.any() else 0.0

        event_mask = np.zeros((len(self.events), len(self.term_names)), dtype=np.uint8)
        name_to_i = {n: i for i, n in enumerate(self.term_names)}
        for t, evs in enumerate(self.events):
            for name in evs:
                if name in name_to_i:
                    event_mask[t, name_to_i[name]] = 1

        with h5py.File(hdf5_path, "w") as handle:
            handle.create_dataset(
                "obs/image_scene",
                data=np.asarray(self.images_scene, dtype=np.uint8),
                compression="gzip",
                compression_opts=1,
            )
            handle.create_dataset(
                "obs/image_ego",
                data=np.asarray(self.images_ego, dtype=np.uint8),
                compression="gzip",
                compression_opts=1,
            )
            handle.create_dataset("obs/qpos", data=np.asarray(self.qpos, np.float32))
            handle.create_dataset("obs/qvel", data=np.asarray(self.qvel, np.float32))
            handle.create_dataset("obs/ee_pos", data=np.asarray(self.ee_pos, np.float32))
            handle.create_dataset("action", data=np.asarray(self.actions, np.float32))
            handle.create_dataset(
                "phase",
                data=np.asarray(self.phases, dtype="S24"),
            )
            handle.create_dataset("reward", data=rewards)
            handle.create_dataset("reward_terms", data=terms)
            handle.create_dataset("events", data=event_mask)
            handle.attrs["reward_term_names"] = [n.encode() for n in self.term_names]
            handle.attrs["action_joints"] = [n.encode() for n in ACTION_JOINTS]
            handle.attrs["task_id"] = self.task_id
            handle.attrs["episode_id"] = episode_id

        payload = {
            "task_id": self.task_id,
            "episode_id": episode_id,
            "num_steps": self.num_steps,
            "return": float(rewards.sum()) if self.num_steps else 0.0,
            "phase_returns": phase_returns,
            **meta,
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return payload
