"""Scripted abort (approach then retract) and hold-zero policies."""

import numpy as np


class BoundaryPolicy:
    """Drives Task1HandController internals without running pick-place."""

    def __init__(self, controller, kind):
        self.controller = controller
        self.kind = kind
        if kind == "kinematics_limit":
            self.phase = "hold"
            self.action_mode = "hold_zero"
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
        cube = ctrl.data.xpos[ctrl.cube_id].copy()
        site = ctrl.data.site_xpos[ctrl.site_id].copy()

        if self.kind == "kinematics_limit":
            self.phase = "hold"
            ctrl._set_fingers_open()
            self._freeze_target()
            ctrl._slew_to_ik(dt)
            ctrl._apply_pd(closing_fingers=False)
            if self.phase_time > 2.5:
                self.done = True
            return

        ctrl._set_fingers_open()
        if self.phase == "approach_side":
            approach = np.array(
                [cube[0] - 0.08, cube[1], max(site[2], ctrl.clear_z)]
            )
            ctrl._track(approach, 0.08, dt)
            near = float(np.linalg.norm(site[:2] - approach[:2])) < 0.04
            if near or self.phase_time > 1.4:
                self.phase = "abort_retract"
                self.phase_time = 0.0
                print("[Task1Collect] approach blocked, retracting")
        else:
            self.phase = "abort_retract"
            retract = np.array(
                [
                    min(site[0], cube[0]) - 0.14,
                    site[1],
                    ctrl.clear_z + 0.04,
                ]
            )
            ctrl._track(retract, 0.08, dt)
            if self.phase_time > 1.6:
                self._freeze_target()
                self.done = True

        ctrl._apply_pd(closing_fingers=False)
