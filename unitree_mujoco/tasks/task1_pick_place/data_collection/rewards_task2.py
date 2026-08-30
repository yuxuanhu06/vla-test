"""Per-step shaped rewards for Task 2 identify-and-pick."""

import numpy as np

from .schema import REWARD_CLIP
from .schema_task2 import REWARD_TERMS


ONCE_EVENTS = {
    "approach_correct",
    "grasp_correct",
    "lift_clear",
    "carry_toward_pad",
    "place_centered",
    "withdraw_correct",
    "safe_abort",
    "identify_correct",
    "neighbor_clear",
    "knock_neighbor",
    "identify_wrong",
    "select_wrong",
    "wrong_place",
}


class RewardTracker:
    def __init__(self, table_top, cube_half, pad_half):
        self.table_top = float(table_top)
        self.cube_half = float(cube_half)
        self.pad_half = float(pad_half)
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
        cube = np.asarray(snapshot["cube_pos"], float)
        pad = np.asarray(snapshot["pad_pos"], float)
        cube_vel = np.asarray(snapshot["cube_vel"], float)
        phase = snapshot["phase"]
        holding = bool(snapshot["holding"])
        firm = bool(snapshot["grip_firm"])
        fingers_closing = bool(snapshot["fingers_closing"])
        hand_hits_cube = bool(snapshot["hand_hits_cube"])
        cube_on_pad = bool(snapshot["cube_on_pad"])
        tipped = bool(snapshot["cube_tipped"])
        action = np.asarray(snapshot["action"], float)
        action_norm = float(np.linalg.norm(action))
        decision_ok = bool(snapshot.get("decision_ok", True))
        holding_wrong = bool(snapshot.get("holding_wrong", False))
        neighbors_disturbed = bool(snapshot.get("neighbors_disturbed", False))
        other_on_pad = bool(snapshot.get("other_on_pad", False))
        required_bound = bool(snapshot.get("required_bound", False))

        to_pad_xy = (pad - cube)[:2]
        pad_dist = float(np.linalg.norm(to_pad_xy) + 1e-9)
        toward_pad = float(cube_vel[:2] @ (to_pad_xy / pad_dist))

        if self.prev is not None:
            site_delta = site - np.asarray(self.prev["ee_pos"], float)
        else:
            site_delta = np.zeros(3)

        pre_grasp = np.array([cube[0] - 0.12, cube[1], cube[2]])
        approaching_pre = float(site_delta @ (pre_grasp - site)) > 1e-5
        attempting = episode_kind not in ("kinematics_limit",)

        if required_bound and decision_ok and "identify_correct" not in self.fired:
            if phase not in ("receive", "understand"):
                terms["identify_correct"] = 1.0
                events.append("identify_correct")

        if required_bound and not decision_ok:
            terms["identify_wrong"] = -1.0
            events.append("identify_wrong")

        staged_side = (
            site[0] < cube[0] - 0.06
            and abs(site[1] - cube[1]) < 0.08
            and not fingers_closing
        )
        if attempting and episode_kind != "tipped" and "approach_correct" not in self.fired and (
            (staged_side and approaching_pre)
            or phase in ("descend_side", "insert")
        ):
            terms["approach_correct"] = 1.0
            events.append("approach_correct")
        if episode_kind == "tipped" and "approach_correct" not in self.fired and (
            staged_side or phase in ("descend_side", "insert", "grasp")
        ):
            terms["approach_correct"] = 1.0
            events.append("approach_correct")

        if (
            attempting
            and episode_kind != "tipped"
            and "grasp_correct" not in self.fired
            and (
                (holding and firm and not tipped)
                or (holding and phase in ("squeeze", "lift_check"))
            )
        ):
            terms["grasp_correct"] = 1.0
            events.append("grasp_correct")

        if (
            attempting
            and "lift_clear" not in self.fired
            and holding
            and cube[2] > self.table_top + self.cube_half + 0.02
        ):
            terms["lift_clear"] = 1.0
            events.append("lift_clear")

        if "carry_toward_pad" not in self.fired and holding and toward_pad > 0.04:
            terms["carry_toward_pad"] = 1.0
            events.append("carry_toward_pad")

        if (
            "place_centered" not in self.fired
            and phase in ("lower", "release", "verify_place", "withdraw", "done")
            and pad_dist < 0.025
        ):
            terms["place_centered"] = 1.0
            events.append("place_centered")

        if (
            "withdraw_correct" not in self.fired
            and phase in ("verify_place", "withdraw", "done", "abort_retract")
            and cube_on_pad
            and site_delta[0] < -0.001
        ):
            terms["withdraw_correct"] = 1.0
            events.append("withdraw_correct")

        if (
            episode_kind in ("space_constraint", "tipped", "wrong_object")
            and "safe_abort" not in self.fired
            and phase == "abort_retract"
        ):
            terms["safe_abort"] = 1.0
            events.append("safe_abort")

        if (
            "neighbor_clear" not in self.fired
            and phase in ("lift_check", "move_above_target", "lower", "done")
            and holding
            and not neighbors_disturbed
        ):
            terms["neighbor_clear"] = 1.0
            events.append("neighbor_clear")

        if site[0] > cube[0] + 0.03 and float(site_delta @ (cube - site)) > 1e-4:
            terms["approach_wrong"] = -1.0
            events.append("approach_wrong")
        if site[2] > cube[2] + 0.07 and fingers_closing and not holding:
            terms["approach_wrong"] = -1.0
            events.append("approach_wrong")

        if fingers_closing and not firm and phase in ("grasp", "squeeze"):
            if snapshot.get("phase_time", 0.0) > 1.0:
                terms["grasp_wrong"] = -1.0
                events.append("grasp_wrong")
        if episode_kind == "tipped" and fingers_closing:
            terms["grasp_wrong"] = -1.0
            events.append("grasp_wrong")

        if holding and toward_pad < -0.04:
            terms["carry_wrong"] = -1.0
            events.append("carry_wrong")

        if hand_hits_cube and phase in (
            "open",
            "rise",
            "approach_side",
            "descend_side",
        ):
            terms["knock"] = -1.0
            events.append("knock")

        if neighbors_disturbed:
            terms["knock_neighbor"] = -1.0
            events.append("knock_neighbor")

        if holding_wrong:
            terms["select_wrong"] = -1.0
            events.append("select_wrong")

        if other_on_pad:
            terms["wrong_place"] = -1.0
            events.append("wrong_place")

        if self.was_holding and not holding and phase not in (
            "release",
            "verify_place",
            "withdraw",
            "done",
            "hold",
        ):
            terms["drop"] = -1.0
            events.append("drop")

        if episode_kind == "kinematics_limit" and action_norm > 0.02:
            terms["illegal_motion"] = -1.0
            events.append("illegal_motion")

        if not holding and phase in (
            "rise",
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
