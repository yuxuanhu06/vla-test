"""Scripted abort, hold-zero, and lying-stick try policies for Task 3."""

import numpy as np


class BoundaryPolicy:
    """Drives Task3HandController internals without running the full skill."""

    def __init__(self, controller, kind):
        self.controller = controller
        self.kind = kind
        if kind == "too_far":
            self.phase = "hold"
            self.action_mode = "hold_zero"
        elif kind == "stick_lying":
            self.phase = "approach_side"
            self.action_mode = "lying_try"
        else:
            self.phase = "approach_side"
            self.action_mode = "retract"
        self.phase_time = 0.0
        self.done = False
        self._freeze_target()

    def _freeze_target(self):
        ctrl = self.controller
        ctrl.q_target[ctrl.ik_acts] = ctrl.data.qpos[ctrl.ik_qadr]
        ctrl.q_ik = ctrl.data.qpos[ctrl.ik_qadr].copy()
        site = ctrl.data.site_xpos[ctrl.site_id].copy()
        ctrl.goal = site

    def step(self, dt):
        ctrl = self.controller
        self.phase_time += dt
        site = ctrl.data.site_xpos[ctrl.site_id].copy()
        stick = ctrl.data.xpos[ctrl.stick_id].copy()
        clear_z = getattr(ctrl, "clear_z", ctrl.table_top + 0.18)

        if self.kind == "too_far":
            self.phase = "hold"
            ctrl.phase = "hold"
            ctrl._set_fingers_open()
            self._freeze_target()
            ctrl._slew_to_ik(dt)
            ctrl._apply_pd(closing_fingers=False)
            if self.phase_time > 2.5:
                self.done = True
            return

        if self.kind == "stick_lying":
            if self.phase == "approach_side":
                ctrl._set_fingers_open()
                approach = np.array(
                    [stick[0] - 0.10, stick[1], max(site[2], clear_z)]
                )
                ctrl._track(approach, 0.08, dt)
                near = float(np.linalg.norm(site[:2] - approach[:2])) < 0.05
                if near or self.phase_time > 1.6:
                    self.phase = "grasp"
                    self.phase_time = 0.0
            elif self.phase == "grasp":
                insert = np.array([stick[0] - 0.02, stick[1], stick[2] + 0.01])
                ctrl._track(insert, 0.06, dt)
                for act_id, value in ctrl.finger_close_target.items():
                    ctrl.q_target[act_id] = value
                if self.phase_time > 1.2:
                    self.phase = "abort_retract"
                    self.phase_time = 0.0
                    print("[Task3Collect] stick is lying, aborting without recover")
            else:
                self.phase = "abort_retract"
                ctrl._set_fingers_open()
                retract = np.array(
                    [min(site[0], stick[0]) - 0.14, site[1], clear_z + 0.04]
                )
                ctrl._track(retract, 0.08, dt)
                if self.phase_time > 1.4:
                    self._freeze_target()
                    self.done = True
            ctrl.phase = self.phase
            ctrl._apply_pd(closing_fingers=self.phase == "grasp")
            return

        ctrl._set_fingers_open()
        if self.phase == "approach_side":
            approach = np.array(
                [stick[0] - 0.08, stick[1], max(site[2], clear_z)]
            )
            ctrl._track(approach, 0.08, dt)
            near = float(np.linalg.norm(site[:2] - approach[:2])) < 0.04
            if near or self.phase_time > 1.4:
                self.phase = "abort_retract"
                self.phase_time = 0.0
                print("[Task3Collect] approach blocked, retracting")
        else:
            self.phase = "abort_retract"
            retract = np.array(
                [min(site[0], stick[0]) - 0.14, site[1], clear_z + 0.04]
            )
            ctrl._track(retract, 0.08, dt)
            if self.phase_time > 1.6:
                self._freeze_target()
                self.done = True
        ctrl.phase = self.phase
        ctrl._apply_pd(closing_fingers=False)
