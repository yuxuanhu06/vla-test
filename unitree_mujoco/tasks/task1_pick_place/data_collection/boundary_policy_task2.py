"""Scripted abort, hold-zero, wrong-object, and tipped-try policies."""

import numpy as np

from task2_hand_controller import Task2HandController


class BoundaryPolicy:
    """Drives Task2HandController internals without running pick-place."""

    def __init__(self, controller, kind, neighbor_type=None):
        self.controller = controller
        self.kind = kind
        self.neighbor_type = neighbor_type
        self.phase = "hold" if kind == "kinematics_limit" else "approach_side"
        self.action_mode = {
            "kinematics_limit": "hold_zero",
            "tipped": "tipped_try",
            "wrong_object": "retract",
        }.get(kind, "retract")
        self.phase_time = 0.0
        self.done = False
        self._target_id = None
        self._bound = False

    def _body_id(self, typ):
        name = Task2HandController.OBJECT_TYPES[typ][0]
        return self.controller.model.body(name).id

    def _ensure_bound(self):
        ctrl = self.controller
        if ctrl.cube_id is None:
            return False
        if self.kind == "wrong_object":
            # Requested type is parked off-table. Do not bind it or a neighbor.
            self._target_id = None
            return False
        self._target_id = ctrl.cube_id
        if not self._bound:
            if ctrl.cube_geom_id is None:
                item = {
                    "body": ctrl.model.body(ctrl.cube_id).name,
                    "geom": [
                        g
                        for t, (_b, g) in Task2HandController.OBJECT_TYPES.items()
                        if ctrl.model.body(_b).id == ctrl.cube_id
                    ][0],
                }
                ctrl._bind_target(item)
            self._bound = True
        return True

    def _freeze_target(self):
        ctrl = self.controller
        ctrl.q_target[ctrl.ik_acts] = ctrl.data.qpos[ctrl.ik_qadr]
        ctrl.q_ik = ctrl.data.qpos[ctrl.ik_qadr].copy()
        site = ctrl.data.site_xpos[ctrl.site_id].copy()
        ctrl.goal = site

    def _object_pos(self):
        return self.controller.data.xpos[self._target_id].copy()

    def step(self, dt):
        ctrl = self.controller
        self.phase_time += dt
        if self.kind == "wrong_object":
            ctrl._set_fingers_open()
            # Approach the occupied row, then abort — requested type is absent.
            if self.phase == "approach_side":
                approach = np.array([0.28, 0.22, ctrl.table_top + 0.18])
                ctrl._track(approach, 0.08, dt)
                site = ctrl.data.site_xpos[ctrl.site_id].copy()
                if float(np.linalg.norm(site[:2] - approach[:2])) < 0.05 or self.phase_time > 1.6:
                    self.phase = "abort_retract"
                    self.phase_time = 0.0
                    print("[Task2Collect] requested object is missing, aborting")
            else:
                self.phase = "abort_retract"
                site = ctrl.data.site_xpos[ctrl.site_id].copy()
                retract = np.array([site[0] - 0.12, site[1], ctrl.table_top + 0.22])
                ctrl._track(retract, 0.08, dt)
                if self.phase_time > 1.4:
                    self._freeze_target()
                    self.done = True
            ctrl.phase = self.phase
            ctrl._apply_pd(closing_fingers=False)
            return

        if not self._ensure_bound():
            ctrl._set_fingers_open()
            self._freeze_target()
            ctrl._slew_to_ik(dt)
            ctrl._apply_pd(closing_fingers=False)
            return

        site = ctrl.data.site_xpos[ctrl.site_id].copy()
        cube = self._object_pos()
        clear_z = getattr(ctrl, "clear_z", ctrl.table_top + 0.185)

        if self.kind == "kinematics_limit":
            self.phase = "hold"
            ctrl.phase = "hold"
            ctrl._set_fingers_open()
            self._freeze_target()
            ctrl._slew_to_ik(dt)
            ctrl._apply_pd(closing_fingers=False)
            if self.phase_time > 2.5:
                self.done = True
            return

        if self.kind == "tipped":
            if self.phase == "approach_side":
                ctrl._set_fingers_open()
                approach = np.array([cube[0] - 0.10, cube[1], max(site[2], clear_z)])
                ctrl._track(approach, 0.08, dt)
                near = float(np.linalg.norm(site[:2] - approach[:2])) < 0.05
                if near or self.phase_time > 1.6:
                    self.phase = "grasp"
                    self.phase_time = 0.0
            elif self.phase == "grasp":
                insert = np.array([cube[0] - 0.02, cube[1], cube[2] + 0.01])
                ctrl._track(insert, 0.06, dt)
                for act_id, value in ctrl.finger_close_target.items():
                    ctrl.q_target[act_id] = value
                if self.phase_time > 1.2:
                    self.phase = "abort_retract"
                    self.phase_time = 0.0
                    print("[Task2Collect] tipped object, aborting without respawn")
            else:
                self.phase = "abort_retract"
                ctrl._set_fingers_open()
                retract = np.array(
                    [min(site[0], cube[0]) - 0.14, site[1], clear_z + 0.04]
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
                [cube[0] - 0.08, cube[1], max(site[2], clear_z)]
            )
            ctrl._track(approach, 0.08, dt)
            near = float(np.linalg.norm(site[:2] - approach[:2])) < 0.04
            if near or self.phase_time > 1.4:
                self.phase = "abort_retract"
                self.phase_time = 0.0
                print("[Task2Collect] approach blocked or wrong object, retracting")
        else:
            self.phase = "abort_retract"
            retract = np.array(
                [min(site[0], cube[0]) - 0.14, site[1], clear_z + 0.04]
            )
            ctrl._track(retract, 0.08, dt)
            if self.phase_time > 1.6:
                self._freeze_target()
                self.done = True
        ctrl.phase = self.phase
        ctrl._apply_pd(closing_fingers=False)
