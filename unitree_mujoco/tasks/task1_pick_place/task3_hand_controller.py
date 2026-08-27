import numpy as np
import mujoco


class Task3HandController:
    """Pick up the standing stick, rotate it in the air, push with the end face.

    The stick is never laid on the table to make it horizontal. The hand must
    not drive the cube; only the held stick's circular end face may push it.
    """

    _SCENE_CONTACT = None

    IK_JOINTS = (
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    )

    REST_POSE = np.array([0.0, 0.30, 0.40, 0.0, 0.60, 0.0, -1.10, 0.0])
    # Palm-up plant: do not pull wrist_pitch back to the table-grasp rest pose.
    STAND_REST = np.array([0.0, 0.20, 0.50, 0.0, 0.80, 0.0, 0.0, 0.0])
    JOINT_WEIGHTS = np.array([6.0, 1.0, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5])

    FINGER_CLOSE = {
        "left_hand_thumb_0_joint": 0.55,
        "left_hand_thumb_1_joint": 0.85,
        "left_hand_thumb_2_joint": 1.00,
        "left_hand_index_0_joint": -0.95,
        "left_hand_index_1_joint": -0.80,
        "left_hand_middle_0_joint": -0.95,
        "left_hand_middle_1_joint": -0.80,
    }

    THUMB_BODIES = (
        "left_hand_thumb_0_link",
        "left_hand_thumb_1_link",
        "left_hand_thumb_2_link",
    )
    FINGER_BODIES = (
        "left_hand_index_0_link",
        "left_hand_index_1_link",
        "left_hand_middle_0_link",
        "left_hand_middle_1_link",
    )

    APPROACH_AXIS = np.array([1.0, 0.0, 0.0])
    CUBE_NORMAL = np.array([0.0, 1.0, 0.0])
    STICK_AXIS_PUSH = np.array([0.0, -1.0, 0.0])

    # Palm-up wrap (vertical stick): fingers +x, jaw +y, palm +z.
    R_GRASP = np.eye(3)
    # Palm facing -y, fingers +x, jaw +z: stick axis along -y, fully level.
    _push_x = np.array([1.0, 0.0, 0.0])
    _push_z = np.array([0.0, -1.0, 0.0])
    _push_y = np.cross(_push_z, _push_x)
    R_PUSH = np.column_stack((_push_x, _push_y, _push_z))

    START_POSE = {
        "left_shoulder_pitch_joint": -0.60,
        "left_shoulder_roll_joint": 0.40,
        "left_elbow_joint": 1.00,
        "left_wrist_pitch_joint": -1.20,
        "right_shoulder_pitch_joint": 0.60,
        "right_shoulder_roll_joint": -0.20,
        "right_elbow_joint": 0.90,
    }

    HOLDING_PHASES = (
        "grasp",
        "squeeze",
        "lift",
        "rotate",
        "lower",
        "approach_cube",
        "press",
        "push",
        "retreat",
        "reorient",
        "center",
        "seat",
    )

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ik_data = mujoco.MjData(model)

        self.site_id = model.site("left_grasp_site").id
        self.target_id = model.body("green_target").id
        pad_geom_id = model.geom("green_target_geom").id
        self.pad_half = float(model.geom_size[pad_geom_id][0])

        self.stick_id = model.body("tool_stick").id
        self.stick_geom_id = model.geom("tool_stick_geom").id
        self.stick_radius = float(model.geom_size[self.stick_geom_id][0])
        self.stick_half = float(model.geom_size[self.stick_geom_id][1])
        stick_jnt = model.body_jntadr[self.stick_id]
        self.stick_qadr = model.jnt_qposadr[stick_jnt]
        self.stick_vadr = model.jnt_dofadr[stick_jnt]

        self.cube_id = model.body("obj_cube").id
        self.cube_geom_id = model.geom("obj_cube_geom").id
        self.cube_half = float(model.geom_size[self.cube_geom_id][2])
        cube_jnt = model.body_jntadr[self.cube_id]
        self.cube_qadr = model.jnt_qposadr[cube_jnt]
        self.cube_vadr = model.jnt_dofadr[cube_jnt]

        table_geom = model.geom("table_top").id
        self.table_top = float(
            data.geom_xpos[table_geom][2] + model.geom_size[table_geom][2]
        )

        self.act_of_joint = {}
        for act_id in range(model.nu):
            joint_id = model.actuator_trnid[act_id, 0]
            self.act_of_joint[model.joint(joint_id).name] = act_id

        self.qadr = np.zeros(model.nu, dtype=int)
        self.vadr = np.zeros(model.nu, dtype=int)
        for act_id in range(model.nu):
            joint_id = model.actuator_trnid[act_id, 0]
            self.qadr[act_id] = model.jnt_qposadr[joint_id]
            self.vadr[act_id] = model.jnt_dofadr[joint_id]

        joint_ids = [model.joint(n).id for n in self.IK_JOINTS]
        self.ik_acts = np.array(
            [self.act_of_joint[n] for n in self.IK_JOINTS], dtype=int
        )
        self.ik_qadr = np.array(
            [model.jnt_qposadr[j] for j in joint_ids], dtype=int
        )
        self.ik_dofs = np.array(
            [model.jnt_dofadr[j] for j in joint_ids], dtype=int
        )
        self.ik_lo = np.array([model.jnt_range[j][0] for j in joint_ids])
        self.ik_hi = np.array([model.jnt_range[j][1] for j in joint_ids])
        self.waist_cap = 0.50
        waist = self.IK_JOINTS.index("waist_yaw_joint")
        self.ik_lo[waist] = max(self.ik_lo[waist], -self.waist_cap)
        self.ik_hi[waist] = min(self.ik_hi[waist], self.waist_cap)

        self.finger_acts = []
        self.finger_open_target = {}
        self.finger_close_target = {}
        for name, act_id in self.act_of_joint.items():
            if not name.startswith("left_hand_"):
                continue
            self.finger_acts.append(act_id)
            joint_id = model.joint(name).id
            lo, hi = model.jnt_range[joint_id]
            self.finger_open_target[act_id] = 0.0
            if name in self.FINGER_CLOSE:
                self.finger_close_target[act_id] = float(
                    np.clip(self.FINGER_CLOSE[name], lo, hi)
                )
        self.finger_acts = np.array(sorted(self.finger_acts), dtype=int)

        self.hand_geoms = self._geoms_of(self.THUMB_BODIES + self.FINGER_BODIES)
        for body_name in ("left_hand_palm_link", "left_wrist_yaw_link"):
            try:
                body_id = model.body(body_name).id
            except KeyError:
                continue
            start = model.body_geomadr[body_id]
            for geom_id in range(start, start + model.body_geomnum[body_id]):
                self.hand_geoms.add(int(geom_id))
        saved = Task3HandController._SCENE_CONTACT
        if saved is None:
            saved = {
                "contype": {
                    gid: int(model.geom_contype[gid]) for gid in self.hand_geoms
                },
                "conaffinity": {
                    gid: int(model.geom_conaffinity[gid])
                    for gid in self.hand_geoms
                },
                "stick_friction": model.geom_friction[self.stick_geom_id].copy(),
                "cube_friction": model.geom_friction[self.cube_geom_id].copy(),
            }
            Task3HandController._SCENE_CONTACT = saved
        self._hand_contype = saved["contype"]
        self._hand_conaffinity = saved["conaffinity"]
        self._stick_friction = saved["stick_friction"]
        self._cube_friction = saved["cube_friction"]
        self._set_hand_collisions(True)

        for name, value in self.START_POSE.items():
            joint_id = model.joint(name).id
            lo, hi = model.jnt_range[joint_id]
            data.qpos[model.jnt_qposadr[joint_id]] = np.clip(value, lo, hi)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        self.q_home = data.qpos[self.qadr].copy()
        self.q_target = self.q_home.copy()

        self.ik_iters = 10
        self.ik_damping = 1.0e-4
        self.max_ik_step = 0.03
        self.ori_weight = 0.7
        self.roll_weight = 1.0
        self.null_gain = 0.4
        self.limit_gain = 0.15
        self.limit_slack = 0.15
        self.max_joint_rate = 1.8

        self.kp_arm = 120.0
        self.kd_arm = 8.0
        self.kp_hold = 150.0
        self.kd_hold = 10.0
        self.kp_finger = 3.0
        self.kd_finger = 0.25
        self.kp_finger_close = 12.0
        self.kd_finger_close = 0.35

        self.approach_offset = 0.12
        self.grasp_dz = 0.0
        self.grasp_inset = 0.012
        self.clear_z = self.table_top + 0.18
        self.air_z = self.table_top + self.stick_half + 0.10
        self.work_z = float(data.xpos[self.cube_id][2])
        self.park_pose = np.array([0.20, 0.26, self.clear_z])
        self.stick_spawn = data.xpos[self.stick_id].copy()

        self.speed_fast = 0.16
        self.speed_slow = 0.05
        self.push_speed = 0.05
        self.pos_tol_approach = 0.03
        self.pos_tol_close = 0.018
        self.tilt_tol = 8.0
        self.max_attempts = 5
        self.ik_stall_time = 0.0
        self.lost_grip_time = 0.0
        self.grip_lost_debounce = 0.25

        self.ori_mode = "grasp"
        self.grip_offset = np.array([self.grasp_inset, 0.0, -self.grasp_dz])
        self.grip_rot = None
        self.lifted = False
        self.rotated_in_air = False
        self.face_contact = False
        self.used_stick = False
        self.succeeded = False
        self.lift_xy = None
        self.place_xy = None
        self.press_z = None
        self.plant_seeded = False
        self.plant_seed_key = None
        self.plant_align = False
        self.seat_xy = None
        self.seat_locked = False
        self.seat_frozen = False
        self.seat_lower_time = 0.0
        self.withdraw_xy = None

        self.q_ik = self.q_home[self.ik_acts].copy()
        self._start_at_level_pose()
        self.goal = self._site_pos_for(self.q_ik)
        self.setpoint_done = True

        pad = self._pad_pos()
        print(
            f"[Task3Hand] pick-rotate-push: pad at "
            f"[{pad[0]:.3f} {pad[1]:.3f} {pad[2]:.3f}], "
            f"stick {2.0*self.stick_half*100:.0f}cm @ y={data.xpos[self.stick_id][1]:.3f}, "
            f"cube {2.0*self.cube_half*100:.0f}cm @ y={data.xpos[self.cube_id][1]:.3f}"
        )

        self.phase = "approach_side"
        self.phase_time = 0.0
        self.dwell = 0.0
        self.attempts = 0
        self.print_timer = 0.0
        self.log_period = 1.0
        self._set_fingers_open()

    def _pad_pos(self):
        return self.data.xpos[self.target_id].copy()

    def _geoms_of(self, body_names):
        wanted = set()
        for name in body_names:
            try:
                body_id = self.model.body(name).id
            except KeyError:
                continue
            start = self.model.body_geomadr[body_id]
            for geom_id in range(start, start + self.model.body_geomnum[body_id]):
                wanted.add(int(geom_id))
        return wanted

    def _target_rot(self):
        return self.R_PUSH if self.ori_mode == "push" else self.R_GRASP

    def _apply_ik_pose(self, goal):
        q, residual, _ = self._solve_ik(goal)
        if residual > 0.03:
            print(f"[Task3Hand] IK pose off by {residual:.3f} m")
            return False
        self.q_ik = q
        self.data.qpos[self.ik_qadr] = q
        self.data.qvel[self.ik_dofs] = 0.0
        for act_id, value in zip(self.ik_acts, q):
            self.q_target[act_id] = value
        mujoco.mj_forward(self.model, self.data)
        self.goal = self._site_pos_for(self.q_ik)
        return True

    def _start_at_level_pose(self):
        block = np.array([0.34, 0.26, self.table_top + 0.055])
        grasp = np.array([block[0] - 0.02, block[1], block[2] + 0.01])
        pre_grasp = np.array([grasp[0] - 0.12, grasp[1], grasp[2]])
        clear_z = self.table_top + 2.0 * 0.055 + 0.075
        self._apply_ik_pose(np.array([pre_grasp[0], pre_grasp[1], clear_z]))

    def _site_pose(self):
        return (
            self.data.site_xpos[self.site_id].copy(),
            self.data.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    def _site_pos_for(self, q_ik):
        scratch = self.ik_data
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self.ik_qadr] = q_ik
        mujoco.mj_kinematics(self.model, scratch)
        return scratch.site_xpos[self.site_id].copy()

    def _tilt_deg(self, rot):
        if self.ori_mode == "stand":
            palm_z = float(np.clip(rot[2, 2], -1.0, 1.0))
            return float(np.degrees(np.arccos(palm_z)))
        if self.ori_mode == "push":
            err = self._ori_error(rot)
            return float(np.degrees(np.linalg.norm(err)))
        aligned = float(np.clip(rot[:, 0] @ self.APPROACH_AXIS, -1.0, 1.0))
        return float(np.degrees(np.arccos(aligned)))

    def _limit_push(self, q, lo=None, hi=None):
        lo = self.ik_lo if lo is None else lo
        hi = self.ik_hi if hi is None else hi
        span = np.maximum(hi - lo, 1e-6)
        slack = self.limit_slack * span
        from_lo = q - lo
        from_hi = hi - q
        push = np.where(from_lo < slack, (slack - from_lo) / slack, 0.0)
        push -= np.where(from_hi < slack, (slack - from_hi) / slack, 0.0)
        return self.limit_gain * push

    def _ori_error(self, rot):
        if self.ori_mode == "push":
            rdes = self._target_rot()
            return 0.5 * (
                np.cross(rot[:, 0], rdes[:, 0])
                + np.cross(rot[:, 1], rdes[:, 1])
                + np.cross(rot[:, 2], rdes[:, 2])
            )
        if self.ori_mode == "stand":
            # Palm +z keeps the hand level and the stick vertical. Wrist yaw
            # is only weakly held so the arm can still translate over the cube.
            palm_up = np.cross(rot[:, 2], np.array([0.0, 0.0, 1.0]))
            yaw = 0.15 * np.cross(rot[:, 0], np.array([1.0, 0.0, 0.0]))
            return palm_up + yaw
        finger_axis = rot[:, 0]
        jaw_axis = rot[:, 1]
        palm_axis = rot[:, 2]
        align = np.cross(finger_axis, self.APPROACH_AXIS)
        flip = 1.0 if palm_axis[2] >= 0.0 else -1.0
        roll = -self.roll_weight * flip * jaw_axis[2] * finger_axis
        return align + roll

    def _ik_track(self, q, goal, iters):
        scratch = self.ik_data
        scratch.qpos[:] = self.data.qpos
        q = q.copy()
        eye6 = np.eye(6)
        eye_n = np.eye(len(q))
        w = self.JOINT_WEIGHTS.copy()
        rest = self.REST_POSE
        lo = self.ik_lo.copy()
        hi = self.ik_hi.copy()
        if self.ori_mode == "stand":
            w[0] = 1.0
            w[2] = 0.7
            rest = self.STAND_REST
            lo[0] = min(lo[0], -0.70)
            hi[0] = max(hi[0], 0.70)
        w_inv = np.diag(1.0 / w)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        if self.ori_mode in ("push", "stand"):
            weight = 2.0
        else:
            weight = self.ori_weight
        for _ in range(iters):
            scratch.qpos[self.ik_qadr] = q
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)
            pos = scratch.site_xpos[self.site_id]
            rot = scratch.site_xmat[self.site_id].reshape(3, 3)
            error = np.concatenate((goal - pos, weight * self._ori_error(rot)))
            mujoco.mj_jacSite(self.model, scratch, jacp, jacr, self.site_id)
            jac = np.vstack((jacp, jacr))[:, self.ik_dofs]
            damped = jac @ w_inv @ jac.T + self.ik_damping * eye6
            dq = w_inv @ jac.T @ np.linalg.solve(damped, error)
            jac_pinv = w_inv @ jac.T @ np.linalg.inv(damped)
            bias = self.null_gain * (rest - q) + self._limit_push(q, lo, hi)
            dq += (eye_n - jac_pinv @ jac) @ bias
            step = float(np.linalg.norm(dq))
            if step > self.max_ik_step:
                dq *= self.max_ik_step / step
            q = np.clip(q + dq, lo, hi)
        return q

    def _solve_ik(self, goal, seed=None, iters=800):
        seed = self.REST_POSE if seed is None else seed
        q = self._ik_track(np.clip(seed, self.ik_lo, self.ik_hi), goal, iters)
        scratch = self.ik_data
        scratch.qpos[self.ik_qadr] = q
        mujoco.mj_kinematics(self.model, scratch)
        pos = scratch.site_xpos[self.site_id].copy()
        rot = scratch.site_xmat[self.site_id].reshape(3, 3)
        return q, float(np.linalg.norm(goal - pos)), self._tilt_deg(rot)

    def _backed_off(self, point, distance):
        return point - distance * self.APPROACH_AXIS

    def _at_clearance(self, point):
        return np.array([point[0], point[1], self.clear_z])

    def _grasp_point(self, stick):
        """Jaw centre at the stick mid-shaft, approached from -x.

        Grasping the geometric centre leaves ~stick_half of shaft between the
        hand and the cube, so the fingers do not reach the cube while pushing.
        """
        return self._backed_off(
            np.array([stick[0], stick[1], stick[2] + self.grasp_dz]),
            self.grasp_inset,
        )

    def _measure_grip_offset(self):
        site_pos, rot = self._site_pose()
        stick = self.data.xpos[self.stick_id]
        self.grip_offset = rot.T @ (stick - site_pos)
        self.grip_rot = rot.copy()

    def _site_goal_from_offset(self, stick_target):
        """Site pose that moves the stick COM to stick_target.

        Use the live hand-to-stick world offset so a stale grasp rotation cannot
        send the site to an unreachable goal while the stick is still held.
        """
        site_pos, _ = self._site_pose()
        stick = self.data.xpos[self.stick_id]
        return site_pos + (np.asarray(stick_target, float) - stick)

    def _stick_axis(self):
        return self.data.xmat[self.stick_id].reshape(3, 3)[:, 2].copy()

    def _stick_axis_toward_cube(self):
        axis = self._stick_axis()
        if axis @ self.STICK_AXIS_PUSH < 0.0:
            axis = -axis
        return axis

    def _stick_end_toward_cube(self):
        stick = self.data.xpos[self.stick_id]
        return stick + self.stick_half * self._stick_axis_toward_cube()

    def _stick_lowest_z(self):
        axis = self._stick_axis()
        az = abs(float(axis[2]))
        radial = float(np.sqrt(max(0.0, 1.0 - az * az)))
        return float(
            self.data.xpos[self.stick_id][2]
            - az * self.stick_half
            - self.stick_radius * radial
        )

    def _stick_clear_of_table(self):
        return self._stick_lowest_z() > self.table_top + 0.008

    def _stick_lifted(self):
        stick = self.data.xpos[self.stick_id]
        return bool(stick[2] > self.table_top + self.stick_half + 0.04)

    def _stick_vertical(self):
        axis = self._stick_axis()
        return abs(float(axis[2])) > 0.97

    def _stick_strictly_vertical(self):
        axis = self._stick_axis()
        return abs(float(axis[2])) > 0.99

    def _stick_vertical_in_air(self):
        return bool(self._stick_vertical() and self._stick_clear_of_table())

    def _cube_top(self):
        return float(self.data.xpos[self.cube_id][2] + self.cube_half)

    def _hover_above_cube_z(self):
        return self._cube_top() + self.stick_half + 0.04

    def _stand_on_cube_z(self):
        return self._cube_top() + self.stick_half

    def _stick_clear_of_cube(self):
        return self._stick_lowest_z() > self._cube_top() + 0.008

    def _stick_standing_on_cube(self):
        stick = self.data.xpos[self.stick_id]
        cube = self.data.xpos[self.cube_id]
        on_top = abs(float(stick[2]) - self._stand_on_cube_z()) < 0.02
        centered = float(np.linalg.norm(stick[:2] - cube[:2])) < self.cube_half
        return bool(self._stick_strictly_vertical() and on_top and centered)

    def _stick_horizontal_in_air(self):
        axis = self._stick_axis_toward_cube()
        aligned = float(axis @ self.STICK_AXIS_PUSH)
        return bool(
            aligned > 0.97
            and abs(float(axis[2])) < 0.10
            and self._stick_clear_of_table()
        )

    def _stick_standing_on_table(self):
        stick = self.data.xpos[self.stick_id]
        axis = self._stick_axis()
        return bool(
            abs(float(axis[2])) > 0.80
            and self.table_top - 0.02 < stick[2] < self.table_top + self.stick_half + 0.08
            and 0.12 < stick[0] < 0.88
            and -0.38 < stick[1] < 0.38
        )

    def _face_flat_on_cube(self):
        if not self._stick_hits_cube():
            return False
        axis = self._stick_axis_toward_cube()
        if float(axis @ self.STICK_AXIS_PUSH) < 0.95:
            return False
        end = self._stick_end_toward_cube()
        cube = self.data.xpos[self.cube_id]
        cube_top = cube[2] + self.cube_half
        cube_bot = cube[2] - self.cube_half
        on_face = (
            cube_bot + self.stick_radius < end[2] < cube_top - self.stick_radius
        )
        centered = abs(end[0] - cube[0]) + self.stick_radius < self.cube_half
        return bool(on_face and centered)

    def _stick_held(self):
        site = self._site_pose()[0]
        stick = self.data.xpos[self.stick_id]
        if np.linalg.norm(stick - site) > 0.08:
            return False
        return self._hand_hits_stick()

    def _grip_lost(self, dt):
        if self._stick_held():
            self.lost_grip_time = 0.0
            return False
        self.lost_grip_time += dt
        return self.lost_grip_time > self.grip_lost_debounce

    def _advance_goal(self, desired, speed, dt):
        delta = desired - self.goal
        distance = float(np.linalg.norm(delta))
        limit = speed * dt
        if distance <= limit or distance < 1e-9:
            self.goal = desired.copy()
            self.setpoint_done = True
        else:
            self.goal = self.goal + delta * (limit / distance)
            self.setpoint_done = False

    def _track(self, desired, speed, dt):
        self._advance_goal(desired, speed, dt)
        self.q_ik = self._ik_track(self.q_ik, self.goal, self.ik_iters)
        residual = np.linalg.norm(self.goal - self._site_pos_for(self.q_ik))
        if residual > 0.02:
            self.ik_stall_time += dt
            if self.ik_stall_time > 0.5:
                self.ik_stall_time = 0.0
                fresh, fresh_residual, _ = self._solve_ik(
                    self.goal, seed=self.q_ik, iters=400
                )
                if fresh_residual < residual:
                    self.q_ik = fresh
        else:
            self.ik_stall_time = 0.0
        self._slew_to_ik(dt)
        site_pos, rot = self._site_pose()
        return float(np.linalg.norm(desired - site_pos)), self._tilt_deg(rot)

    def _hold(self, dt):
        self.q_ik = self._ik_track(self.q_ik, self.goal, self.ik_iters)
        self._slew_to_ik(dt)
        site_pos, rot = self._site_pose()
        return float(np.linalg.norm(self.goal - site_pos)), self._tilt_deg(rot)

    def _slew_to_ik(self, dt):
        limit = self.max_joint_rate * dt
        current = self.q_target[self.ik_acts]
        delta = np.clip(self.q_ik - current, -limit, limit)
        self.q_target[self.ik_acts] = current + delta

    def _freeze_arm(self, dt):
        """Hold the current arm pose. No IK, so contact cannot shake the wrist."""
        if not self.seat_frozen:
            self.q_ik = self.data.qpos[self.ik_qadr].copy()
            self.goal = self._site_pose()[0].copy()
            self.seat_frozen = True
        self._slew_to_ik(dt)
        site_pos, rot = self._site_pose()
        return float(np.linalg.norm(self.goal - site_pos)), self._tilt_deg(rot)

    def _lower_z_only(self, stick_target, speed, dt):
        return self._move_site_wrist_frozen(
            self._site_goal_from_offset(stick_target), speed, dt, 0.2, 1.0
        )

    def _slide_xy_wrist_frozen(self, stick_target, speed, dt):
        return self._move_site_wrist_frozen(
            self._site_goal_from_offset(stick_target), speed, dt, 1.0, 0.4
        )

    def _move_site_wrist_frozen(
        self, site_target, speed, dt, xy_gain=1.0, z_gain=1.0, freeze_yaw=False
    ):
        """Move the grasp site with wrist roll/pitch frozen."""
        self._advance_goal(np.asarray(site_target, float), speed, dt)

        scratch = self.ik_data
        scratch.qpos[:] = self.data.qpos
        q = self.q_ik.copy()
        scratch.qpos[self.ik_qadr] = q
        mujoco.mj_kinematics(self.model, scratch)
        mujoco.mj_comPos(self.model, scratch)
        pos = scratch.site_xpos[self.site_id]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, scratch, jacp, jacr, self.site_id)
        jac = jacp[:, self.ik_dofs].copy()
        jac[:, 5:7] = 0.0
        if freeze_yaw:
            jac[:, 7] = 0.0
        err = self.goal - pos
        err[0] *= xy_gain
        err[1] *= xy_gain
        err[2] *= z_gain
        damped = jac @ jac.T + 1.0e-3 * np.eye(3)
        dq = jac.T @ np.linalg.solve(damped, err)
        dq[5:7] = 0.0
        if freeze_yaw:
            dq[7] = 0.0
        step = float(np.linalg.norm(dq))
        if step > self.max_ik_step:
            dq *= self.max_ik_step / step
        lo = self.ik_lo.copy()
        hi = self.ik_hi.copy()
        lo[0] = min(lo[0], -0.70)
        hi[0] = max(hi[0], 0.70)
        self.q_ik = np.clip(q + dq, lo, hi)
        self._slew_to_ik(dt)
        site_pos, rot = self._site_pose()
        return float(np.linalg.norm(self.goal - site_pos)), self._tilt_deg(rot)

    def _reached(self, pos_error, tilt, tol):
        ori_tol = 18.0 if self.ori_mode in ("push", "stand") else self.tilt_tol
        return self.setpoint_done and pos_error < tol and tilt < ori_tol

    def _set_fingers_open(self):
        for act_id in self.finger_acts:
            self.q_target[act_id] = self.finger_open_target[act_id]

    def _set_fingers_open_staged(self, t):
        """Peel index/middle first; keep the thumb as a backstop, then open it."""
        for name, act_id in self.act_of_joint.items():
            if act_id not in self.finger_open_target:
                continue
            opened = self.finger_open_target[act_id]
            closed = self.finger_close_target.get(act_id, opened)
            if "thumb" in name:
                alpha = float(np.clip((t - 1.8) / 1.2, 0.0, 1.0))
            else:
                alpha = float(np.clip(t / 1.8, 0.0, 1.0))
            self.q_target[act_id] = closed + alpha * (opened - closed)

    def _stick_still(self):
        lin = self.data.qvel[self.stick_vadr : self.stick_vadr + 3]
        ang = self.data.qvel[self.stick_vadr + 3 : self.stick_vadr + 6]
        return float(np.linalg.norm(lin)) < 0.04 and float(np.linalg.norm(ang)) < 0.25

    def _fingers_near_open(self):
        for act_id in self.finger_acts:
            opened = self.finger_open_target[act_id]
            q = float(self.data.qpos[self.qadr[act_id]])
            if abs(q - opened) > 0.18:
                return False
        return True

    def _set_fingers_closed(self):
        for act_id, value in self.finger_close_target.items():
            self.q_target[act_id] = value

    def _pair_hits(self, geom_id, other_set):
        for i in range(self.data.ncon):
            pair = {int(self.data.contact[i].geom1), int(self.data.contact[i].geom2)}
            if geom_id in pair and pair & other_set:
                return True
        return False

    def _hand_hits_stick(self):
        return self._pair_hits(self.stick_geom_id, self.hand_geoms)

    def _set_hand_collisions(self, enabled):
        """Enable or disable hand collisions after the stick is planted.

        A 20 cm / 3 cm cylinder on the cube tips at ~8 deg. Unwrapping the
        Dex3 jaw at mid-shaft always exceeds that, so once the stick is seated
        the fingers leave through a collision-free peel.
        """
        if enabled:
            for gid, value in self._hand_contype.items():
                self.model.geom_contype[gid] = value
            for gid, value in self._hand_conaffinity.items():
                self.model.geom_conaffinity[gid] = value
            self.model.geom_friction[self.stick_geom_id] = self._stick_friction
            self.model.geom_friction[self.cube_geom_id] = self._cube_friction
            self._hand_collisions_off = False
        else:
            for gid in self.hand_geoms:
                self.model.geom_contype[gid] = 0
                self.model.geom_conaffinity[gid] = 0
            self.model.geom_friction[self.stick_geom_id] = np.array(
                [2.4, 0.12, 0.02]
            )
            self.model.geom_friction[self.cube_geom_id] = np.array(
                [2.4, 0.12, 0.02]
            )
            self._hand_collisions_off = True

    def _hand_hits_cube(self):
        return self._pair_hits(self.cube_geom_id, self.hand_geoms)

    def _stick_hits_cube(self):
        for i in range(self.data.ncon):
            pair = {int(self.data.contact[i].geom1), int(self.data.contact[i].geom2)}
            if self.stick_geom_id in pair and self.cube_geom_id in pair:
                return True
        return False

    def _cube_speed(self):
        return float(
            np.linalg.norm(self.data.qvel[self.cube_vadr : self.cube_vadr + 3])
        )

    def _cube_corners_in_pad(self):
        """True if every XY corner of the cube is inside the green square."""
        cube = self.data.xpos[self.cube_id]
        pad = self._pad_pos()
        rot = self.data.xmat[self.cube_id].reshape(3, 3)
        limit = self.pad_half - 0.002
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                corner = cube + rot @ np.array(
                    [sx * self.cube_half, sy * self.cube_half, 0.0]
                )
                if abs(float(corner[0] - pad[0])) > limit:
                    return False
                if abs(float(corner[1] - pad[1])) > limit:
                    return False
        return True

    def _cube_on_pad(self):
        cube = self.data.xpos[self.cube_id]
        resting = (
            self.table_top - 0.01 < cube[2] < self.table_top + self.cube_half + 0.04
        )
        return bool(
            self._cube_corners_in_pad()
            and resting
            and self._cube_speed() < 0.12
        )

    def _clip_torque(self, act_id, torque):
        if self.model.actuator_forcelimited[act_id]:
            lo, hi = self.model.actuator_forcerange[act_id]
            if lo < hi:
                return float(np.clip(torque, lo, hi))
        if self.model.actuator_ctrllimited[act_id]:
            lo, hi = self.model.actuator_ctrlrange[act_id]
            if lo < hi:
                return float(np.clip(torque, lo, hi))
        joint_id = self.model.actuator_trnid[act_id, 0]
        lo, hi = self.model.jnt_actfrcrange[joint_id]
        if lo < hi:
            return float(np.clip(torque, lo, hi))
        return float(torque)

    def _apply_pd(self, closing_fingers):
        finger_set = set(self.finger_acts.tolist())
        ik_set = set(self.ik_acts.tolist())
        for act_id in range(self.model.nu):
            q = self.data.qpos[self.qadr[act_id]]
            dq = self.data.qvel[self.vadr[act_id]]
            if act_id in finger_set:
                kp = self.kp_finger_close if closing_fingers else self.kp_finger
                kd = self.kd_finger_close if closing_fingers else self.kd_finger
                gravity = 0.0
            else:
                if act_id in ik_set:
                    kp, kd = self.kp_arm, self.kd_arm
                else:
                    kp, kd = self.kp_hold, self.kd_hold
                joint_id = self.model.actuator_trnid[act_id, 0]
                lo, hi = self.model.jnt_actfrcrange[joint_id]
                bias = self.data.qfrc_bias[self.vadr[act_id]]
                gravity = float(np.clip(bias, lo, hi)) if lo < hi else bias
            torque = kp * (self.q_target[act_id] - q) - kd * dq + gravity
            self.data.ctrl[act_id] = self._clip_torque(act_id, torque)

    def _set_phase(self, phase):
        if phase != self.phase:
            print(f"[Task3Hand] {self.phase} -> {phase}")
        self.phase = phase
        self.phase_time = 0.0
        self.dwell = 0.0
        site_pos, _ = self._site_pose()
        self.goal = site_pos
        self.setpoint_done = True
        if phase == "press":
            self.press_z = float(self.data.xpos[self.stick_id][2])

    def _begin_recover(self, reason):
        print(f"[Task3Hand] {reason}; recovering (attempt {self.attempts + 1})")
        self.attempts += 1
        self.lifted = False
        self.rotated_in_air = False
        self.face_contact = False
        self.lift_xy = None
        self.place_xy = None
        self.lost_grip_time = 0.0
        self.grip_rot = None
        self.press_z = None
        self.plant_seeded = False
        self.plant_seed_key = None
        self.plant_align = False
        self.seat_xy = None
        self.seat_locked = False
        self.seat_frozen = False
        self.seat_lower_time = 0.0
        self.withdraw_xy = None
        self._set_hand_collisions(True)
        self._set_fingers_open()
        self._set_phase("recover")

    def _maybe_drop(self, dt):
        if self.phase in self.HOLDING_PHASES and self._grip_lost(dt):
            if self.succeeded:
                print("[Task3Hand] lost the stick after success")
                self._set_fingers_open()
                self._set_phase("done")
                return True
            self._begin_recover("dropped the stick")
            return True
        return False

    def step(self):
        dt = self.model.opt.timestep
        self.phase_time += dt
        self.print_timer += dt

        stick = self.data.xpos[self.stick_id].copy()
        cube = self.data.xpos[self.cube_id].copy()
        pad = self._pad_pos()
        grasp = self._grasp_point(stick)
        pre = self._backed_off(grasp, self.approach_offset)
        travel = self._at_clearance(pre)
        site_pos, _ = self._site_pose()

        closing = self.phase in self.HOLDING_PHASES
        pos_error = 0.0
        tilt = 0.0

        if self.phase == "approach_side":
            self.ori_mode = "grasp"
            self._set_fingers_open()
            if not self._stick_standing_on_table():
                print("[Task3Hand] stick is not standing on the table")
                self._set_phase("done")
            else:
                pos_error, tilt = self._track(travel, self.speed_fast, dt)
                if self._hand_hits_cube():
                    self._begin_recover("hand touched the cube")
                elif self._reached(pos_error, tilt, self.pos_tol_approach):
                    self._set_phase("descend_side")
                elif self.phase_time > 15.0:
                    self._begin_recover("could not reach the stick staging pose")

        elif self.phase == "descend_side":
            self._set_fingers_open()
            pos_error, tilt = self._track(pre, self.speed_slow, dt)
            if self._hand_hits_cube():
                self._begin_recover("hand touched the cube")
            elif self._reached(pos_error, tilt, self.pos_tol_close):
                self._set_phase("insert")
            elif self.phase_time > 12.0:
                self._begin_recover("could not drop beside the stick")

        elif self.phase == "insert":
            self._set_fingers_open()
            pos_error, tilt = self._track(grasp, self.speed_slow, dt)
            if self._hand_hits_cube():
                self._begin_recover("hand touched the cube")
            elif self._reached(pos_error, tilt, self.pos_tol_close):
                print("[Task3Hand] stick in the jaw, closing fingers")
                self._set_phase("grasp")
            elif self.phase_time > 12.0:
                self._begin_recover("could not slide onto the stick")

        elif self.phase == "grasp":
            self._set_fingers_closed()
            pos_error, tilt = self._track(grasp, self.speed_slow, dt)
            if self._stick_held():
                self.dwell += dt
                if self.dwell > 0.3:
                    self._set_phase("squeeze")
            elif self.phase_time > 4.0:
                self._begin_recover("fingers closed but found no grip")

        elif self.phase == "squeeze":
            self._set_fingers_closed()
            pos_error, tilt = self._hold(dt)
            if self._maybe_drop(dt):
                pass
            elif self.phase_time > 0.7:
                self._measure_grip_offset()
                print("[Task3Hand] lifting the stick off the table")
                self._set_phase("lift")

        elif self.phase == "lift":
            self._set_fingers_closed()
            lift_stick = np.array([stick[0], stick[1], self.air_z])
            if self.lift_xy is None:
                self.lift_xy = stick[:2].copy()
                lift_stick[:2] = self.lift_xy
            else:
                lift_stick[:2] = self.lift_xy
            pos_error, tilt = self._track(
                self._site_goal_from_offset(lift_stick), self.speed_slow, dt
            )
            if self._maybe_drop(dt):
                pass
            elif self._hand_hits_cube():
                self._begin_recover("hand touched the cube")
            elif self._stick_lifted() and self._stick_clear_of_table():
                self.dwell += dt
                if self.dwell > 0.35:
                    self.lifted = True
                    self._measure_grip_offset()
                    print("[Task3Hand] stick in the air, rotating to horizontal")
                    self._set_phase("rotate")
            elif self.phase_time > 12.0:
                self._begin_recover("could not lift the stick off the table")

        elif self.phase == "rotate":
            self._set_fingers_closed()
            self.ori_mode = "push"
            hold = np.array(
                [
                    self.lift_xy[0] if self.lift_xy is not None else stick[0],
                    self.lift_xy[1] if self.lift_xy is not None else stick[1],
                    self.air_z,
                ]
            )
            if not self._stick_clear_of_table():
                hold[2] = self.air_z + 0.06
            pos_error, tilt = self._track(
                self._site_goal_from_offset(hold), self.speed_slow, dt
            )
            if self._maybe_drop(dt):
                pass
            elif not self._stick_clear_of_table() and self.phase_time > 1.0:
                self.air_z = min(self.air_z + 0.04, self.table_top + self.stick_half + 0.22)
                print(f"[Task3Hand] stick dipped; raising air height to {self.air_z:.2f}")
            elif self._stick_horizontal_in_air():
                self.dwell += dt
                if self.dwell > 0.4:
                    self.rotated_in_air = True
                    self._measure_grip_offset()
                    print("[Task3Hand] stick horizontal in the air")
                    self._set_phase("lower")
            elif self.phase_time > 10.0:
                self._begin_recover("could not rotate the stick in the air")

        elif self.phase == "lower":
            self._set_fingers_closed()
            self.ori_mode = "push"
            if not self._stick_clear_of_table() and not self._stick_horizontal_in_air():
                self._begin_recover("stick touched the table while horizontal")
            else:
                work = np.array(
                    [
                        self.lift_xy[0] if self.lift_xy is not None else cube[0],
                        self.lift_xy[1] if self.lift_xy is not None else stick[1],
                        self.work_z,
                    ]
                )
                pos_error, tilt = self._track(
                    self._site_goal_from_offset(work), self.speed_slow, dt
                )
                if self._maybe_drop(dt):
                    pass
                elif self._hand_hits_cube():
                    self._begin_recover("hand touched the cube")
                elif stick[2] <= self.work_z + 0.012 and self._stick_horizontal_in_air():
                    self.dwell += dt
                    if self.dwell > 0.2:
                        print("[Task3Hand] translating toward the cube")
                        self._set_phase("press")
                elif (
                    pos_error < 0.03
                    and self._stick_horizontal_in_air()
                    and self.phase_time > 3.0
                ):
                    print("[Task3Hand] translating toward the cube")
                    self._set_phase("press")
                elif self.phase_time > 10.0:
                    if self._stick_horizontal_in_air() and self._stick_clear_of_table():
                        print("[Task3Hand] lower timed out, seating at current height")
                        self._set_phase("press")
                    else:
                        self._begin_recover("could not lower to cube height")

        elif self.phase == "approach_cube":
            self._set_phase("press")

        elif self.phase == "press":
            self._set_fingers_closed()
            self.ori_mode = "push"
            desired_stick = np.array(
                [
                    cube[0],
                    cube[1] + self.cube_half + self.stick_half - 0.008,
                    stick[2] if self.press_z is None else self.press_z,
                ]
            )
            pos_error, tilt = self._track(
                self._site_goal_from_offset(desired_stick), 0.03, dt
            )
            if self._maybe_drop(dt):
                pass
            elif self._hand_hits_cube() and not self._stick_hits_cube():
                self._begin_recover("hand touched the cube; must push with the stick")
            elif self._face_flat_on_cube() or (
                self._stick_hits_cube()
                and float(self._stick_axis_toward_cube() @ self.STICK_AXIS_PUSH) > 0.97
                and abs(self._stick_end_toward_cube()[0] - cube[0]) < 0.03
            ):
                self.dwell += dt
                if self.dwell > 0.35:
                    self.face_contact = True
                    self.used_stick = True
                    print("[Task3Hand] full end-face contact, pushing to the pad")
                    self._set_phase("push")
            elif self.phase_time > 16.0:
                self._begin_recover("could not seat the stick face on the cube")

        elif self.phase == "push":
            self._set_fingers_closed()
            self.ori_mode = "push"
            if not (self.lifted and self.rotated_in_air and self.face_contact):
                self._begin_recover("push gated until lift, in-air rotate, and face contact")
            elif self._hand_hits_cube() and not self._stick_hits_cube() and not self.used_stick:
                self._begin_recover("hand touched the cube; must push with the stick")
            elif self._maybe_drop(dt):
                pass
            else:
                desired_stick = np.array(
                    [
                        cube[0],
                        cube[1] + self.cube_half + self.stick_half - 0.05,
                        stick[2] if self.press_z is None else self.press_z,
                    ]
                )
                if self._face_flat_on_cube() or self._stick_hits_cube():
                    self.used_stick = True
                if self._cube_corners_in_pad() and self.used_stick:
                    self.dwell += dt
                    pos_error, tilt = self._hold(dt)
                    if self.dwell > 0.25 and (
                        self._cube_on_pad() or self.dwell > 0.55
                    ):
                        print("[Task3Hand] success: stick used, cube fully on pad")
                        self.succeeded = True
                        self.place_xy = None
                        print("[Task3Hand] lifting the stick")
                        self._set_phase("retreat")
                else:
                    self.dwell = 0.0
                    pos_error, tilt = self._track(
                        self._site_goal_from_offset(desired_stick),
                        self.push_speed,
                        dt,
                    )
                    if self.phase_time > 45.0:
                        self._begin_recover("cube did not reach the pad")

        elif self.phase == "retreat":
            self._set_fingers_closed()
            self.ori_mode = "push"
            if self.place_xy is None:
                self.place_xy = stick[:2].copy()
            hover_z = self._hover_above_cube_z()
            hold = np.array([self.place_xy[0], self.place_xy[1], hover_z])
            pos_error, tilt = self._track(
                self._site_goal_from_offset(hold), self.speed_slow, dt
            )
            if self._maybe_drop(dt):
                pass
            elif self._stick_clear_of_cube() and stick[2] >= hover_z - 0.03:
                self.dwell += dt
                if self.dwell > 0.2:
                    self._measure_grip_offset()
                    print("[Task3Hand] rotating the stick to vertical in place")
                    self._set_phase("reorient")
            elif self.phase_time > 10.0:
                self._measure_grip_offset()
                print("[Task3Hand] rotating the stick to vertical in place")
                self._set_phase("reorient")

        elif self.phase == "reorient":
            self._set_fingers_closed()
            self.ori_mode = "stand"
            hover_z = self._hover_above_cube_z()
            hold = np.array(
                [
                    self.place_xy[0] if self.place_xy is not None else stick[0],
                    self.place_xy[1] if self.place_xy is not None else stick[1],
                    hover_z,
                ]
            )
            if not self._stick_clear_of_cube():
                hold[2] = hover_z + 0.06
            pos_error, tilt = self._track(
                self._site_goal_from_offset(hold), self.speed_slow, dt
            )
            if self._maybe_drop(dt):
                pass
            elif self._stick_strictly_vertical() and self._stick_clear_of_cube():
                self.dwell += dt
                if self.dwell > 0.35:
                    self._measure_grip_offset()
                    print("[Task3Hand] stick vertical, sliding in the air to cube center")
                    self._set_phase("center")
            elif self.phase_time > 12.0:
                self._measure_grip_offset()
                print("[Task3Hand] sliding in the air to cube center")
                self._set_phase("center")

        elif self.phase == "center":
            self._set_fingers_closed()
            self.ori_mode = "stand"
            hover_z = self._hover_above_cube_z()
            if not self.plant_seeded:
                self.q_ik = self.data.qpos[self.ik_qadr].copy()
                self.plant_seeded = True
            target = np.array([cube[0], cube[1], hover_z])
            if not self._stick_clear_of_cube() or self._stick_hits_cube():
                target[2] = hover_z + 0.06
            pos_error, tilt = self._slide_xy_wrist_frozen(target, 0.04, dt)
            xy_err = float(np.linalg.norm(stick[:2] - cube[:2]))
            on_center = (
                xy_err < 0.015
                and self._stick_strictly_vertical()
                and stick[2] >= hover_z - 0.04
                and self._stick_clear_of_cube()
            )
            if self._maybe_drop(dt):
                pass
            elif on_center:
                self.dwell += dt
                if self.dwell > 0.3:
                    print("[Task3Hand] stopped above cube center, lowering")
                    self._set_phase("seat")
            elif self.phase_time > 22.0 and xy_err < 0.015 and self._stick_strictly_vertical():
                print("[Task3Hand] stopped above cube center, lowering")
                self._set_phase("seat")

        elif self.phase == "seat":
            self._set_fingers_closed()
            self.ori_mode = "stand"
            hover_z = self._hover_above_cube_z()
            stand_z = self._stand_on_cube_z()
            xy_err = float(np.linalg.norm(stick[:2] - cube[:2]))
            if xy_err < 0.015 and self._stick_strictly_vertical():
                self.seat_locked = True
            touching = self._stick_hits_cube() or (
                self._stick_lowest_z() <= self._cube_top() + 0.006
            )
            if touching and self._stick_strictly_vertical():
                pos_error, tilt = self._freeze_arm(dt)
                if self._maybe_drop(dt):
                    pass
                elif self._stick_still():
                    self.dwell += dt
                    if self.dwell > 0.5:
                        print("[Task3Hand] stick touching the cube, opening the hand")
                        self._set_phase("release")
            elif not self._stick_strictly_vertical():
                self.seat_locked = False
                target = np.array([cube[0], cube[1], hover_z])
                pos_error, tilt = self._track(
                    self._site_goal_from_offset(target), 0.03, dt
                )
                if self._maybe_drop(dt):
                    pass
            elif not self.seat_locked:
                target = np.array([cube[0], cube[1], hover_z])
                pos_error, tilt = self._track(
                    self._site_goal_from_offset(target), 0.04, dt
                )
                if self._maybe_drop(dt):
                    pass
            else:
                self.seat_lower_time += dt
                target = np.array([cube[0], cube[1], stand_z])
                pos_error, tilt = self._lower_z_only(target, 0.03, dt)
                if self._maybe_drop(dt):
                    pass
                elif self.seat_lower_time > 8.0:
                    print("[Task3Hand] lowering timed out, opening the hand")
                    self._set_phase("release")

        elif self.phase == "release":
            self.ori_mode = "stand"
            pos_error, tilt = self._freeze_arm(dt)
            if self.phase_time < 0.35:
                self._set_fingers_closed()
            else:
                if not self._hand_collisions_off:
                    self._set_hand_collisions(False)
                self._set_fingers_open_staged(self.phase_time - 0.35)
            if self.phase_time > 2.4:
                print("[Task3Hand] stick standing on the cube, withdrawing the arm")
                self._set_phase("withdraw")

        elif self.phase == "withdraw":
            self._set_fingers_open()
            self.ori_mode = "stand"
            if self.withdraw_xy is None:
                self.withdraw_xy = site_pos.copy()
                self.q_ik = self.data.qpos[self.ik_qadr].copy()
                self.seat_frozen = False
                print(
                    "[Task3Hand] sliding the hand backward off the stick "
                    f"(open={int(self._fingers_near_open())} "
                    f"hit={int(self._hand_hits_stick())})"
                )
            back_x = self.withdraw_xy[0] - 0.14
            lift_z = max(self.withdraw_xy[2] + 0.10, self.clear_z)
            # Pull the jaw off the shaft first. Lifting while the palm is still
            # around the stick knocks it off the cube.
            if site_pos[0] > back_x + 0.03:
                away = np.array(
                    [back_x, self.withdraw_xy[1], self.withdraw_xy[2]]
                )
                pos_error, tilt = self._move_site_wrist_frozen(
                    away, 0.03, dt, xy_gain=1.0, z_gain=0.25, freeze_yaw=True
                )
            elif site_pos[2] < lift_z - 0.03:
                up = np.array([back_x, self.withdraw_xy[1], lift_z])
                pos_error, tilt = self._move_site_wrist_frozen(
                    up, 0.04, dt, xy_gain=0.35, z_gain=1.0, freeze_yaw=True
                )
            else:
                pos_error, tilt = self._track(self.park_pose, self.speed_slow, dt)
                if self._reached(pos_error, tilt, self.pos_tol_approach):
                    print("[Task3Hand] arm clear")
                    self._set_phase("done")
            if self.phase == "withdraw" and self.phase_time > 12.0:
                print("[Task3Hand] arm clear")
                self._set_phase("done")

        elif self.phase == "recover":
            self._set_fingers_open()
            if site_pos[2] < self.clear_z - 0.05:
                lift = np.array([site_pos[0], site_pos[1], self.clear_z])
                pos_error, tilt = self._track(lift, self.speed_fast, dt)
            else:
                self.ori_mode = "grasp"
                pos_error, tilt = self._track(self.park_pose, self.speed_fast, dt)
            if self.attempts >= self.max_attempts:
                print(f"[Task3Hand] giving up after {self.attempts} attempts")
                self._set_phase("done")
            elif self.phase_time > 2.2:
                self._set_phase("approach_side")

        else:
            self._set_fingers_open()
            if self.succeeded:
                if self._hand_collisions_off and not self._hand_hits_stick():
                    self._set_hand_collisions(True)
                self._slew_to_ik(dt)
                site_pos, rot = self._site_pose()
                pos_error = float(np.linalg.norm(self.goal - site_pos))
                tilt = self._tilt_deg(rot)
            else:
                pos_error, tilt = self._hold(dt)

        self._apply_pd(closing_fingers=closing)

        if self.print_timer > self.log_period:
            self.print_timer = 0.0
            axis = self._stick_axis_toward_cube()
            xy_err = float(np.linalg.norm(stick[:2] - cube[:2]))
            print(
                f"[Task3Hand] phase={self.phase:14s} "
                f"pos_err={pos_error:.3f} tilt={tilt:4.1f} "
                f"xy={xy_err*1000:4.0f}mm "
                f"stick={np.round(stick, 3)} cube={np.round(cube, 3)} "
                f"axis_y={axis[1]:+.2f} axis_z={self._stick_axis()[2]:+.2f} "
                f"lowz={self._stick_lowest_z():.3f} "
                f"held={int(self._stick_held())} "
                f"face={int(self._face_flat_on_cube())} "
                f"center={np.linalg.norm(cube[:2]-pad[:2])*1000:4.0f}mm"
            )
