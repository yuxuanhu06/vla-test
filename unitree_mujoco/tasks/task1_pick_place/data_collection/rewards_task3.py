"""Per-step shaped rewards for Task 3 stick-push."""

import numpy as np

from .schema import REWARD_CLIP
from .schema_task3 import REWARD_TERMS


ONCE_EVENTS = {
    "approach_correct",
    "grasp_correct",
    "lift_clear",
    "rotate_in_air",
    "face_contact",
    "push_toward_pad",
    "place_cube",
    "plant_stick",
    "withdraw_correct",
    "safe_abort",
}


class RewardTracker:
    def __init__(self, table_top, cube_half, pad_half, stick_half):
        self.table_top = float(table_top)
        self.cube_half = float(cube_half)
        self.pad_half = float(pad_half)
        self.stick_half = float(stick_half)
        self.fired = set()
        self.was_holding = False
        self.prev = None

    def reset(self):
        self.fired.clear()
        self.was_holding = False
        self.prev = None

    def step(self, snapshot, episode_kind):
        terms = {name: 0.0 for name in REWARD_TERMS}
        events = []

        site = np.asarray(snapshot["ee_pos"], float)
        stick = np.asarray(snapshot["stick_pos"], float)
        cube = np.asarray(snapshot["cube_pos"], float)
        pad = np.asarray(snapshot["pad_pos"], float)
        cube_vel = np.asarray(snapshot["cube_vel"], float)
        phase = snapshot["phase"]
        holding = bool(snapshot["holding"])
        firm = bool(snapshot["grip_firm"])
        fingers_closing = bool(snapshot["fingers_closing"])
        hand_hits_cube = bool(snapshot["hand_hits_cube"])
        cube_on_pad = bool(snapshot["cube_on_pad"])
        stick_on_cube = bool(snapshot["stick_standing_on_cube"])
        stick_lying = bool(snapshot["stick_lying"])
        rotated = bool(snapshot.get("rotated_in_air", False))
        face = bool(snapshot.get("face_contact", False))
        action = np.asarray(snapshot["action"], float)
        action_norm = float(np.linalg.norm(action))

        to_pad_xy = (pad - cube)[:2]
        pad_dist = float(np.linalg.norm(to_pad_xy) + 1e-9)
        toward_pad = float(cube_vel[:2] @ (to_pad_xy / pad_dist))

        if self.prev is not None:
            site_delta = site - np.asarray(self.prev["ee_pos"], float)
        else:
            site_delta = np.zeros(3)

        pre_grasp = np.array([stick[0] - 0.12, stick[1], stick[2]])
        approaching_pre = float(site_delta @ (pre_grasp - site)) > 1e-5
        attempting = episode_kind not in ("too_far",)

        staged_side = (
            site[0] < stick[0] - 0.06
            and abs(site[1] - stick[1]) < 0.08
            and not fingers_closing
        )
        if attempting and "approach_correct" not in self.fired and (
            (staged_side and approaching_pre)
            or phase in ("descend_side", "insert", "grasp")
        ):
            terms["approach_correct"] = 1.0
            events.append("approach_correct")

        if (
            attempting
            and episode_kind != "stick_lying"
            and "grasp_correct" not in self.fired
            and (
                (holding and firm and not stick_lying)
                or (holding and phase in ("squeeze", "lift"))
            )
        ):
            terms["grasp_correct"] = 1.0
            events.append("grasp_correct")

        if (
            attempting
            and "lift_clear" not in self.fired
            and holding
            and stick[2] > self.table_top + self.stick_half + 0.04
        ):
            terms["lift_clear"] = 1.0
            events.append("lift_clear")

        if "rotate_in_air" not in self.fired and (
            rotated or phase in ("rotate", "lower", "press", "push")
        ) and holding:
            terms["rotate_in_air"] = 1.0
            events.append("rotate_in_air")

        if "face_contact" not in self.fired and (
            face or phase in ("push", "retreat")
        ):
            terms["face_contact"] = 1.0
            events.append("face_contact")

        if "push_toward_pad" not in self.fired and holding and toward_pad > 0.03:
            terms["push_toward_pad"] = 1.0
            events.append("push_toward_pad")

        if (
            "place_cube" not in self.fired
            and cube_on_pad
            and phase in ("push", "retreat", "reorient", "center", "seat", "release", "withdraw", "done")
        ):
            terms["place_cube"] = 1.0
            events.append("place_cube")

        if (
            "plant_stick" not in self.fired
            and stick_on_cube
            and phase in ("seat", "release", "withdraw", "done")
        ):
            terms["plant_stick"] = 1.0
            events.append("plant_stick")

        if (
            "withdraw_correct" not in self.fired
            and phase in ("withdraw", "done", "abort_retract")
            and cube_on_pad
            and site_delta[0] < -0.001
        ):
            terms["withdraw_correct"] = 1.0
            events.append("withdraw_correct")

        if (
            episode_kind in ("space_constraint", "stick_lying")
            and "safe_abort" not in self.fired
            and phase == "abort_retract"
        ):
            terms["safe_abort"] = 1.0
            events.append("safe_abort")

        if site[0] > stick[0] + 0.03 and float(site_delta @ (stick - site)) > 1e-4:
            terms["approach_wrong"] = -1.0
            events.append("approach_wrong")

        if fingers_closing and not firm and phase in ("grasp", "squeeze"):
            if snapshot.get("phase_time", 0.0) > 1.0:
                terms["grasp_wrong"] = -1.0
                events.append("grasp_wrong")
        if episode_kind == "stick_lying" and fingers_closing:
            terms["grasp_wrong"] = -1.0
            events.append("grasp_wrong")

        if hand_hits_cube and phase in (
            "approach_side",
            "descend_side",
            "insert",
            "press",
            "push",
        ):
            terms["hand_hit_cube"] = -1.0
            events.append("hand_hit_cube")
            terms["knock"] = -1.0
            events.append("knock")

        if self.was_holding and not holding and phase not in (
            "release",
            "withdraw",
            "done",
            "hold",
            "abort_retract",
        ):
            terms["drop"] = -1.0
            events.append("drop")

        if episode_kind == "too_far" and action_norm > 0.02:
            terms["illegal_motion"] = -1.0
            events.append("illegal_motion")

        if not holding and phase in (
            "approach_side",
            "descend_side",
            "insert",
        ):
            gdir = pre_grasp - site
            gn = float(np.linalg.norm(gdir) + 1e-9)
            if float(site_delta @ (gdir / gn)) > 1e-4:
                terms["dense_toward"] = 0.05
        if holding and toward_pad < -0.02:
            terms["dense_away"] = -0.05
        elif holding and toward_pad > 0.02:
            terms["dense_toward"] = 0.05

        for name in events:
            if name in ONCE_EVENTS:
                self.fired.add(name)

        reward = float(np.clip(sum(terms.values()), -REWARD_CLIP, REWARD_CLIP))
        self.was_holding = holding
        self.prev = snapshot
        return reward, terms, events
