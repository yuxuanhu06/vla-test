import numpy as np
import mujoco


def mat_to_axis_angle(mat):
    """Convert a 3x3 rotation matrix to an axis-angle vector."""
    rot = np.asarray(mat, dtype=float).reshape(3, 3)
    cos = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=float,
    )
    axis = axis / (2.0 * np.sin(angle) + 1e-12)
    return axis * angle


def axis_angle_to_mat(vec):
    """Rodrigues map from an axis-angle vector to a 3x3 rotation matrix."""
    w = np.asarray(vec, dtype=float).reshape(3)
    angle = float(np.linalg.norm(w))
    if angle < 1e-9:
        return np.eye(3)
    axis = w / angle
    kx, ky, kz = axis
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


class Task1HandController:
    """Pick the red cube with the left Dex3 hand and place it on the green pad.

    The arm follows a Cartesian setpoint that a redundancy-resolved IK solver
    converts into joint targets. The solver runs on a scratch MjData seeded from
    its own previous solution rather than from the measured pose, which is what
    keeps it from winding up into the joint limits whenever the PD loop lags.
    """

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

    # Posture the redundant DOFs are pulled towards: arm forward, elbow down.
    REST_POSE = np.array([0.0, 0.30, 0.15, 0.0, 0.60, 0.0, -1.10, 0.0])

    # waist_yaw displaces the palm further per radian than any arm joint, so an
    # unweighted solve spends its motion there and the torso twists instead of
    # the arm extending.
    JOINT_WEIGHTS = np.array([12.0, 1.0, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5])

    # All seven driven hand joints take part: the thumb's three, plus two each
    # for index and middle. The knuckles (joint 0) close hardest so the block is
    # taken deep in the wrap, and the tips (joint 1) only curl enough to conform
    # around it. Curling the tips all the way instead squeezes the block out
    # towards the fingernails, which holds by friction alone. Measured with
    # finger_study.py: this wrap keeps four contact points at about 32 N, where
    # a fingertip pinch reaches 60 N on two points.
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

    # The fingers point forward and the hand slides horizontally onto the block.
    # A top-down grasp is not possible with this hand: the fingertips trail
    # 0.065 m below the palm, so with the palm level they stand on the table
    # long before the jaw reaches an object resting on it. Coming in sideways
    # keeps the whole hand well clear of the table. Rotation about this axis is
    # left free, since it does not change where the jaw closes on a square
    # block, and constraining it wastes a DOF the arm needs for a natural
    # posture.
    APPROACH_AXIS = np.array([1.0, 0.0, 0.0])

    # At qpos=0 both G1 arms stick straight forward at roughly table height, so
    # the hands start inside the table top. The left arm is lifted clear and the
    # right arm is parked at the robot's side; leaving the right hand resting on
    # the table would anchor the torso by friction and block waist_yaw.
    START_POSE = {
        "left_shoulder_pitch_joint": -0.60,
        "left_shoulder_roll_joint": 0.40,
        "left_elbow_joint": 1.00,
        "left_wrist_pitch_joint": -1.20,
        "right_shoulder_pitch_joint": 0.60,
        "right_shoulder_roll_joint": -0.20,
        "right_elbow_joint": 0.90,
    }

    def __init__(self, model, data, expert_init=True):
        self.model = model
        self.data = data
        self.expert_init = bool(expert_init)

        self.ik_data = mujoco.MjData(model)

        self.site_id = model.site("left_grasp_site").id
        self.cube_id = model.body("red_cube").id
        self.cube_geom_id = model.geom("red_cube_geom").id
        self.target_id = model.body("green_target").id
        pad_geom_id = model.geom("green_target_geom").id

        self.cube_half = float(model.geom_size[self.cube_geom_id][2])
        self.pad_half = float(model.geom_size[pad_geom_id][0])
        self.pad_top = float(
            data.xpos[self.target_id][2] + model.geom_size[pad_geom_id][2]
        )

        cube_joint = model.body_jntadr[self.cube_id]
        self.cube_qadr = int(model.jnt_qposadr[cube_joint])
        self.cube_vadr = int(model.jnt_dofadr[cube_joint])
        self.cube_spawn = data.qpos[self.cube_qadr:self.cube_qadr + 7].copy()

        table_geom = model.geom("table_top").id
        self.table_top = float(
            data.geom_xpos[table_geom][2] + model.geom_size[table_geom][2]
        )
        self.table_center = data.geom_xpos[table_geom][:2].copy()
        self.table_half = model.geom_size[table_geom][:2].copy()

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

        # Every waypoint is reachable with under 15 deg of torso rotation, so
        # the waist is clamped well inside its real travel. Left free it becomes
        # the solver's cheapest way out of a tight pose and the robot spins.
        self.waist_cap = 0.35
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

        self.thumb_geoms = self._geoms_of(self.THUMB_BODIES)
        self.finger_geoms = self._geoms_of(self.FINGER_BODIES)

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
        self.max_joint_rate = 1.2

        self.kp_arm = 120.0
        self.kd_arm = 8.0
        self.kp_hold = 150.0
        self.kd_hold = 10.0
        self.kp_finger = 3.0
        self.kd_finger = 0.25
        self.kp_finger_close = 12.0
        self.kd_finger_close = 0.35

        # The jaw closes on the block's middle. The hand stages one approach
        # offset back along -APPROACH_AXIS, then slides straight in.
        self.approach_offset = 0.12
        self.grasp_dz = 0.01
        self.grasp_inset = 0.02
        self.lift_height = 0.10
        self.retreat_height = 0.14

        # Heights of the block itself above the pad, not of the grasp site.
        self.hover_clear = 0.06
        self.place_gap = 0.006

        # Nominal seat of the block in the jaw, refined once the grip is firm.
        self.grip_offset = np.array([self.grasp_inset, 0.0, -self.grasp_dz])

        # Every lateral move happens at this height. The fingers hang 0.065 m
        # below the jaw, so anything lower drags them across the top of the
        # block and knocks it over before the hand is even in position.
        self.clear_z = self.table_top + 2.0 * self.cube_half + 0.075

        self.speed_fast = 0.16
        self.speed_slow = 0.05
        self.lower_speed = 0.025

        self.pos_tol_approach = 0.03
        self.pos_tol_close = 0.015
        self.tilt_tol = 8.0
        self.center_tol = 0.010
        self.grip_force_min = 1.5
        self.grip_lost_debounce = 0.2
        self.max_attempts = 5
        self.ik_stall_time = 0.0
        self.block_goal = None
        self.place_z = None
        self.transport_z = None
        self.grip_rot = None
        self.place_seeded = False

        self.q_ik = self.q_home[self.ik_acts].copy()
        if expert_init:
            self._start_at_travel_pose()
        self.goal = self._site_pos_for(self.q_ik)
        self.setpoint_done = True

        self.phase = "open"
        self.phase_time = 0.0
        self.dwell = 0.0
        self.lost_grip_time = 0.0
        self.settle_time = 0.0
        self.attempts = 0
        self.print_timer = 0.0
        self.log_period = 1.0
        self.pick_pos = data.xpos[self.cube_id].copy()
        self.holding = False
        self.libero_pos_clip = 0.04
        self.libero_ori_clip = 0.20
        self._gripper_alpha = 0.0
        self._libero_closing = False

        self._set_fingers_open()
        if expert_init:
            self._report_feasibility()

    # ------------------------------------------------------------------
    # model helpers
    # ------------------------------------------------------------------

    def _start_at_travel_pose(self):
        """Put the left arm on the travel waypoint before the run begins.

        A hand-picked joint start pose is fragile: the arm only has to be a few
        centimetres off for the palm to rest against the block, and the first
        move then topples it. Starting on the waypoint the arm is about to use
        is clear of the block by construction.
        """
        block = self.data.xpos[self.cube_id].copy()
        pre_grasp = self._backed_off(
            self._grasp_point(block), self.approach_offset
        )
        q, residual, _ = self._solve_ik(self._at_clearance(pre_grasp))
        if residual > 0.03:
            print(
                f"[Task1Hand] travel pose off by {residual:.3f} m, "
                "keeping the default start pose"
            )
            return

        self.q_ik = q
        self.data.qpos[self.ik_qadr] = q
        self.data.qvel[self.ik_dofs] = 0.0
        for act_id, value in zip(self.ik_acts, q):
            self.q_target[act_id] = value
        mujoco.mj_forward(self.model, self.data)

    def _geoms_of(self, body_names):
        wanted = set()
        for name in body_names:
            body_id = self.model.body(name).id
            start = self.model.body_geomadr[body_id]
            for geom_id in range(start, start + self.model.body_geomnum[body_id]):
                wanted.add(int(geom_id))
        return wanted

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
        aligned = float(np.clip(rot[:, 0] @ self.APPROACH_AXIS, -1.0, 1.0))
        return float(np.degrees(np.arccos(aligned)))

    # ------------------------------------------------------------------
    # inverse kinematics
    # ------------------------------------------------------------------

    def _limit_push(self, q):
        span = np.maximum(self.ik_hi - self.ik_lo, 1e-6)
        slack = self.limit_slack * span
        from_lo = q - self.ik_lo
        from_hi = self.ik_hi - q
        push = np.where(from_lo < slack, (slack - from_lo) / slack, 0.0)
        push -= np.where(from_hi < slack, (slack - from_hi) / slack, 0.0)
        return self.limit_gain * push

    def _ori_error(self, rot):
        """Rotation vector driving the hand to a level side grasp.

        Two terms: point the fingers along the approach axis, and keep the jaw
        axis (thumb to fingers, the hand's local y) horizontal so the jaw closes
        across the block's width rather than along its height. Without the
        second term the roll is free and the hand tends to arrive thumb-first,
        which shoves the block instead of taking it in.

        The jaw term is signed by the palm direction, so either hand flip is
        accepted and the arm keeps a full turn of freedom.
        """
        finger_axis = rot[:, 0]
        jaw_axis = rot[:, 1]
        palm_axis = rot[:, 2]

        align = np.cross(finger_axis, self.APPROACH_AXIS)
        flip = 1.0 if palm_axis[2] >= 0.0 else -1.0
        roll = -self.roll_weight * flip * jaw_axis[2] * finger_axis
        return align + roll

    def _ik_track(self, q, goal, iters):
        """Refine q towards goal. Operates purely on the scratch MjData."""
        scratch = self.ik_data
        scratch.qpos[:] = self.data.qpos
        q = q.copy()

        eye6 = np.eye(6)
        eye_n = np.eye(len(q))
        w_inv = np.diag(1.0 / self.JOINT_WEIGHTS)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        for _ in range(iters):
            scratch.qpos[self.ik_qadr] = q
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)

            pos = scratch.site_xpos[self.site_id]
            rot = scratch.site_xmat[self.site_id].reshape(3, 3)
            error = np.concatenate(
                (goal - pos, self.ori_weight * self._ori_error(rot))
            )

            mujoco.mj_jacSite(self.model, scratch, jacp, jacr, self.site_id)
            jac = np.vstack((jacp, jacr))[:, self.ik_dofs]

            damped = jac @ w_inv @ jac.T + self.ik_damping * eye6
            dq = w_inv @ jac.T @ np.linalg.solve(damped, error)

            # The projector has to use the same weighted pseudo-inverse as the
            # task term. With the unweighted one the two disagree and the waist
            # slowly walks away from the posture bias into a contorted solution.
            jac_pinv = w_inv @ jac.T @ np.linalg.inv(damped)
            bias = self.null_gain * (self.REST_POSE - q) + self._limit_push(q)
            dq += (eye_n - jac_pinv @ jac) @ bias

            step = float(np.linalg.norm(dq))
            if step > self.max_ik_step:
                dq *= self.max_ik_step / step

            q = np.clip(q + dq, self.ik_lo, self.ik_hi)

        return q

    def _solve_ik(self, goal, seed=None, iters=800):
        """Converge on goal from scratch. Returns (q, pos_residual, tilt)."""
        seed = self.REST_POSE if seed is None else seed
        q = self._ik_track(np.clip(seed, self.ik_lo, self.ik_hi), goal, iters)

        scratch = self.ik_data
        scratch.qpos[self.ik_qadr] = q
        mujoco.mj_kinematics(self.model, scratch)
        pos = scratch.site_xpos[self.site_id].copy()
        rot = scratch.site_xmat[self.site_id].reshape(3, 3)
        return q, float(np.linalg.norm(goal - pos)), self._tilt_deg(rot)

    def _report_feasibility(self):
        """Fail loudly at startup if a waypoint is out of reach."""
        cube = self.data.xpos[self.cube_id].copy()
        pad = self.data.xpos[self.target_id].copy()
        grasp = self._grasp_point(cube)
        pre_grasp = self._backed_off(grasp, self.approach_offset)
        # The block hangs off the site by grip_offset, so the placing waypoints
        # are the block targets shifted back into site coordinates.
        shift = self.grip_offset * np.array([1.0, 1.0, 1.0])
        waypoints = (
            ("travel", self._at_clearance(pre_grasp)),
            ("pre_grasp", pre_grasp),
            ("grasp", grasp),
            ("lift", self._at_clearance(grasp)),
            (
                "above_target",
                self._block_over_pad(pad, self.hover_clear) - shift,
            ),
            ("place", self._block_over_pad(pad, self.place_gap) - shift),
        )

        print("[Task1Hand] waypoint feasibility (left arm + waist_yaw):")
        seed = self.REST_POSE
        for name, goal in waypoints:
            q, residual, tilt = self._solve_ik(goal, seed=seed)
            flag = ""
            if residual > 0.03:
                flag = "  UNREACHABLE"
            elif tilt > 5.0:
                flag = "  POOR ORIENTATION"
            print(
                f"  {name:13s} goal={np.round(goal, 3)} "
                f"err={residual:.4f} m tilt={tilt:4.1f} deg "
                f"waist_yaw={q[0]:+.3f}{flag}"
            )
            seed = q

    # ------------------------------------------------------------------
    # cartesian waypoints
    # ------------------------------------------------------------------

    def _above(self, point, height):
        return np.array([point[0], point[1], point[2] + height])

    def _backed_off(self, point, distance):
        """Point moved back along the approach axis, i.e. away from the block."""
        return point - distance * self.APPROACH_AXIS

    def _grasp_point(self, block):
        """Where the jaw centre stops: short of the block, not on top of it.

        Driving the site all the way to the block's axis buries it against the
        wrist, which shoves the block out of the jaw. Stopping an inset short
        leaves the block sitting in the jaw with clearance on both sides.
        """
        return self._backed_off(
            self._above(block, self.grasp_dz), self.grasp_inset
        )

    def _at_clearance(self, point):
        """Same horizontal spot, raised to the safe travel height."""
        return np.array([point[0], point[1], self.clear_z])

    def _measure_grip_offset(self):
        """Record where the block sits in the jaw, in the hand's own frame."""
        site_pos, rot = self._site_pose()
        block = self.data.xpos[self.cube_id]
        self.grip_offset = rot.T @ (block - site_pos)
        self.grip_rot = rot.copy()
        self.block_goal = block.copy()

    def _site_goal_from_offset(self, block_target):
        """Fixed site waypoint for a block pose, using the grasp-time offset."""
        rot = self.grip_rot if self.grip_rot is not None else self._site_pose()[1]
        return np.asarray(block_target, float) - rot @ self.grip_offset

    def _site_goal_for_block(self, block_target):
        """Live site goal that puts the held block on block_target."""
        site_pos, _ = self._site_pose()
        block = self.data.xpos[self.cube_id]
        return site_pos + (np.asarray(block_target, float) - block)

    def _block_over_pad(self, pad, height):
        """Where the block should be: on the pad centre, raised by height."""
        return np.array(
            [pad[0], pad[1], self.pad_top + self.cube_half + height]
        )

    def _block_centering_error(self, pad):
        """Horizontal distance from the block to the pad centre."""
        block = self.data.xpos[self.cube_id]
        return float(np.linalg.norm(block[:2] - pad[:2]))

    def _advance_goal_xy_z(self, desired, xy_speed, z_speed, dt):
        """Rate-limit the site setpoint with independent XY and Z speeds.

        A single 3D interpolation spends most of its step on the long axis, so
        a 10 cm descent would starve a 1 cm centering correction and the
        block would drift off the pad while going down.
        """
        desired = np.asarray(desired, float)
        delta = desired - self.goal
        xy = delta[:2]
        dxy = float(np.linalg.norm(xy))
        xy_limit = xy_speed * dt
        if dxy > xy_limit and dxy > 1e-9:
            xy = xy * (xy_limit / dxy)
            xy_done = False
        else:
            xy_done = True
        z_limit = z_speed * dt
        dz = float(np.clip(delta[2], -z_limit, z_limit))
        z_done = abs(delta[2]) <= z_limit + 1e-9
        self.goal = self.goal + np.array([xy[0], xy[1], dz])
        self.setpoint_done = bool(xy_done and z_done)

    def _track_place(self, block_desired, xy_speed, z_speed, dt, live=True):
        """Track a block pose, correcting XY independently of descent."""
        if live:
            desired_site = self._site_goal_for_block(block_desired)
        else:
            desired_site = self._site_goal_from_offset(block_desired)
        self._advance_goal_xy_z(desired_site, xy_speed, z_speed, dt)
        self.q_ik = self._ik_track(self.q_ik, self.goal, self.ik_iters)
        residual = np.linalg.norm(self.goal - self._site_pos_for(self.q_ik))
        if residual > 0.02:
            self.ik_stall_time += dt
            if self.ik_stall_time > 0.5:
                self.ik_stall_time = 0.0
                fresh, fresh_residual, _ = self._solve_ik(self.goal, iters=400)
                if fresh_residual < residual:
                    self.q_ik = fresh
        else:
            self.ik_stall_time = 0.0
        self._slew_to_ik(dt)
        block = self.data.xpos[self.cube_id]
        _, rot = self._site_pose()
        return float(np.linalg.norm(block_desired - block)), self._tilt_deg(rot)

    def _pad_point(self, pad, height):
        return np.array(
            [pad[0], pad[1], self.pad_top + self.cube_half + height]
        )

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

        # If the incremental solve is stuck short of the goal it has fallen into
        # a local minimum, so restart it from the rest posture.
        residual = np.linalg.norm(self.goal - self._site_pos_for(self.q_ik))
        if residual > 0.02:
            self.ik_stall_time += dt
            if self.ik_stall_time > 0.5:
                self.ik_stall_time = 0.0
                fresh, fresh_residual, _ = self._solve_ik(self.goal, iters=400)
                if fresh_residual < residual:
                    print(
                        f"[Task1Hand] IK stalled at {residual:.3f} m, "
                        f"re-solved to {fresh_residual:.3f} m"
                    )
                    self.q_ik = fresh
        else:
            self.ik_stall_time = 0.0

        self._slew_to_ik(dt)
        site_pos, rot = self._site_pose()
        return float(np.linalg.norm(desired - site_pos)), self._tilt_deg(rot)

    def _hold(self, dt):
        """Freeze the arm setpoint, keeping the current IK solution."""
        self._slew_to_ik(dt)
        site_pos, rot = self._site_pose()
        return float(np.linalg.norm(self.goal - site_pos)), self._tilt_deg(rot)

    def _slew_to_ik(self, dt):
        limit = self.max_joint_rate * dt
        current = self.q_target[self.ik_acts]
        delta = np.clip(self.q_ik - current, -limit, limit)
        self.q_target[self.ik_acts] = current + delta

    def _ik_track_pose(self, q, goal_pos, goal_rot, iters):
        """Jacobian IK toward a Cartesian pose. Does not use expert ori priors."""
        scratch = self.ik_data
        scratch.qpos[:] = self.data.qpos
        q = np.asarray(q, dtype=float).copy()
        goal_pos = np.asarray(goal_pos, dtype=float).reshape(3)
        goal_rot = np.asarray(goal_rot, dtype=float).reshape(3, 3)

        eye6 = np.eye(6)
        eye_n = np.eye(len(q))
        w_inv = np.diag(1.0 / self.JOINT_WEIGHTS)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        for _ in range(iters):
            scratch.qpos[self.ik_qadr] = q
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)

            pos = scratch.site_xpos[self.site_id]
            rot = scratch.site_xmat[self.site_id].reshape(3, 3)
            r_err = goal_rot @ rot.T
            error = np.concatenate((goal_pos - pos, mat_to_axis_angle(r_err)))

            mujoco.mj_jacSite(self.model, scratch, jacp, jacr, self.site_id)
            jac = np.vstack((jacp, jacr))[:, self.ik_dofs]

            damped = jac @ w_inv @ jac.T + self.ik_damping * eye6
            dq = w_inv @ jac.T @ np.linalg.solve(damped, error)
            jac_pinv = w_inv @ jac.T @ np.linalg.inv(damped)
            bias = self.null_gain * (self.REST_POSE - q) + self._limit_push(q)
            dq += (eye_n - jac_pinv @ jac) @ bias

            step = float(np.linalg.norm(dq))
            if step > self.max_ik_step:
                dq *= self.max_ik_step / step
            q = np.clip(q + dq, self.ik_lo, self.ik_hi)

        return q

    def ee_state(self):
        """6-D observation.state: site xyz + axis-angle."""
        pos, rot = self._site_pose()
        return np.concatenate([pos, mat_to_axis_angle(rot)]).astype(np.float32)

    def _set_gripper_from_scalar(self, value):
        """Map a LIBERO gripper command in ~[-1, 1] onto the seven Dex3 joints."""
        alpha = float(np.clip((np.clip(value, -1.0, 1.0) + 1.0) * 0.5, 0.0, 1.0))
        self._gripper_alpha = alpha
        self._libero_closing = alpha > 0.35
        for act_id in self.finger_acts:
            opened = self.finger_open_target[act_id]
            closed = self.finger_close_target.get(act_id, opened)
            self.q_target[act_id] = opened + alpha * (closed - opened)

    def set_libero_action(self, action):
        """Apply a 7-D LIBERO action: relative EE pose + gripper. Not step()."""
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size < 7:
            raise ValueError(f"expected 7-D LIBERO action, got shape {action.shape}")
        pos_delta = np.clip(action[:3], -self.libero_pos_clip, self.libero_pos_clip)
        ori_delta = np.clip(action[3:6], -self.libero_ori_clip, self.libero_ori_clip)
        site_pos, site_rot = self._site_pose()
        goal_pos = site_pos + pos_delta
        goal_rot = axis_angle_to_mat(ori_delta) @ site_rot
        self.q_ik = self._ik_track_pose(self.q_ik, goal_pos, goal_rot, self.ik_iters)
        self._set_gripper_from_scalar(float(action[6]))

    def apply_delta_q(self, delta, act_ids, act_qadr, action_clip=0.25, ref_qpos=None):
        """Hold a recorded 15-D joint delta as q_target = qpos + delta (replay only).

        Pass the recorded joint qpos as ref_qpos. Using the live qpos would turn
        a small PD tracking error into a hold-in-place command.
        """
        delta = np.clip(np.asarray(delta, dtype=float).reshape(-1), -action_clip, action_clip)
        if ref_qpos is None:
            ref = self.data.qpos[act_qadr]
        else:
            ref = np.asarray(ref_qpos, dtype=float).reshape(-1)
        if ref.shape[0] != delta.shape[0]:
            raise ValueError(
                f"ref_qpos length {ref.shape[0]} != action length {delta.shape[0]}"
            )
        self.q_target[act_ids] = ref + delta
        self.q_ik = self.q_target[self.ik_acts].copy()
        alphas = []
        for act_id in self.finger_acts:
            opened = self.finger_open_target[act_id]
            closed = self.finger_close_target.get(act_id, opened)
            span = closed - opened
            if abs(span) > 1e-6:
                alphas.append((self.q_target[act_id] - opened) / span)
        self._gripper_alpha = float(np.mean(alphas)) if alphas else 0.0
        self._libero_closing = self._gripper_alpha > 0.35

    def pd_tick(self, slew=True):
        """One PD control tick. Does not run the expert state machine."""
        dt = float(self.model.opt.timestep)
        self.phase_time += dt
        if slew:
            self._slew_to_ik(dt)
        self._apply_pd(bool(self._libero_closing))

    def refresh_hold_flag(self):
        firm = bool(self._grip_is_firm())
        self.holding = firm
        return firm

    def _reached(self, pos_error, tilt, tol):
        return self.setpoint_done and pos_error < tol and tilt < self.tilt_tol

    # ------------------------------------------------------------------
    # gripper
    # ------------------------------------------------------------------

    def _set_fingers_open(self):
        """Drive every Dex3 joint (thumb 3 + index 2 + middle 2) to open."""
        for act_id in self.finger_acts:
            self.q_target[act_id] = self.finger_open_target[act_id]

    def _set_fingers_closed(self):
        for act_id, value in self.finger_close_target.items():
            self.q_target[act_id] = value

    def _fingers_open_error(self):
        """Largest remaining error across the seven hand joints."""
        err = 0.0
        for act_id in self.finger_acts:
            err = max(
                err,
                abs(
                    float(self.q_target[act_id] - self.data.qpos[self.qadr[act_id]])
                ),
            )
        return err

    def _grip_state(self):
        """Count thumb and finger contacts on the cube and sum normal force.

        The palm is deliberately excluded: a thumb-plus-palm push is not a
        grip, but it does produce two contacts.
        """
        thumb = 0
        fingers = 0
        force = 0.0
        wrench = np.zeros(6)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if self.cube_geom_id not in (geom1, geom2):
                continue
            other = geom1 if geom2 == self.cube_geom_id else geom2

            if other in self.thumb_geoms:
                thumb += 1
            elif other in self.finger_geoms:
                fingers += 1
            else:
                continue

            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            force += abs(float(wrench[0]))

        return thumb, fingers, force

    def _grip_is_firm(self):
        thumb, fingers, force = self._grip_state()
        return thumb >= 1 and fingers >= 1 and force >= self.grip_force_min

    def _grip_lost(self, dt):
        """True once the grip has been absent for longer than the debounce."""
        if self._grip_is_firm():
            self.lost_grip_time = 0.0
            return False
        self.lost_grip_time += dt
        return self.lost_grip_time > self.grip_lost_debounce

    # ------------------------------------------------------------------
    # cube bookkeeping
    # ------------------------------------------------------------------

    def _cube_off_table(self):
        cube = self.data.xpos[self.cube_id]
        if cube[2] < self.table_top - 0.05:
            return True
        offset = np.abs(cube[:2] - self.table_center)
        return bool(np.any(offset > self.table_half))

    def _cube_speed(self):
        return float(
            np.linalg.norm(self.data.qvel[self.cube_vadr:self.cube_vadr + 3])
        )

    def _cube_tipped(self):
        """True once the upright block has fallen over.

        The side grasp needs the block standing, so a knocked-over block has to
        be stood back up before another attempt is worth making.
        """
        up = self.data.xmat[self.cube_id].reshape(3, 3)[:, 2]
        return bool(up[2] < 0.8)

    def _cube_on_pad(self):
        """Block counts as placed only if it is near the pad centre.

        The pad is 20 cm across, so a loose footprint check would accept a
        drop well off to one side. Placement is only done once the block is
        sitting close to the painted centre.
        """
        cube = self.data.xpos[self.cube_id]
        pad = self.data.xpos[self.target_id]
        in_xy = float(np.linalg.norm(cube[:2] - pad[:2])) < 0.025
        resting = self.pad_top < cube[2] < self.pad_top + self.cube_half + 0.03
        return bool(in_xy and resting and self._cube_speed() < 0.05)

    def _respawn_cube(self):
        self.data.qpos[self.cube_qadr:self.cube_qadr + 7] = self.cube_spawn
        self.data.qvel[self.cube_vadr:self.cube_vadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        print("[Task1Hand] cube left the table, respawned at its start pose")

    # ------------------------------------------------------------------
    # torque
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # state machine
    # ------------------------------------------------------------------

    def _set_phase(self, phase):
        if phase != self.phase:
            print(f"[Task1Hand] {self.phase} -> {phase}")
        self.phase = phase
        self.phase_time = 0.0
        self.dwell = 0.0
        self.lost_grip_time = 0.0
        self.settle_time = 0.0
        if phase == "move_above_target":
            self.block_goal = self.data.xpos[self.cube_id].copy()
            self.transport_z = float(self.data.xpos[self.cube_id][2])
            self.place_seeded = False
        elif phase == "lower":
            self.place_z = float(self.data.xpos[self.cube_id][2])

    def _begin_recover(self, reason):
        print(f"[Task1Hand] {reason}; recovering (attempt {self.attempts + 1})")
        self.attempts += 1
        self.holding = False
        self._set_fingers_open()
        self._set_phase("recover")

    def step(self):
        dt = self.model.opt.timestep
        self.phase_time += dt
        self.print_timer += dt

        cube = self.data.xpos[self.cube_id].copy()
        pad = self.data.xpos[self.target_id].copy()
        if not self.holding:
            self.pick_pos = cube
        reference = self.pick_pos if self.holding else cube

        grasp_pose = self._grasp_point(reference)
        pre_grasp = self._backed_off(grasp_pose, self.approach_offset)
        site_pos, _ = self._site_pose()
        rise_pose = self._at_clearance(site_pos)
        travel_pose = self._at_clearance(pre_grasp)
        lift_pose = self._at_clearance(self._grasp_point(self.pick_pos))
        hold_z = (
            self.transport_z
            if self.transport_z is not None
            else float(cube[2])
        )
        hover_block = np.array([pad[0], pad[1], hold_z])
        place_block = self._block_over_pad(pad, self.place_gap)
        retreat_pose = self._backed_off(
            self._at_clearance(site_pos), self.retreat_height
        )

        closing = self.phase in (
            "grasp",
            "squeeze",
            "lift_check",
            "move_above_target",
            "lower",
        )

        pos_error = 0.0
        tilt = 0.0

        if self.phase == "open":
            self._set_fingers_open()
            pos_error, tilt = self._hold(dt)
            if self.phase_time > 0.6:
                self._set_phase("rise")

        elif self.phase == "rise":
            # Get the fingertips above the block before travelling sideways.
            self._set_fingers_open()
            pos_error, tilt = self._track(rise_pose, self.speed_slow, dt)
            if self._reached(pos_error, tilt, self.pos_tol_approach):
                self._set_phase("approach_side")
            elif self.phase_time > 8.0:
                self._set_phase("approach_side")

        elif self.phase == "approach_side":
            self._set_fingers_open()
            pos_error, tilt = self._track(travel_pose, self.speed_fast, dt)
            if self._reached(pos_error, tilt, self.pos_tol_approach):
                self._set_phase("descend_side")
            elif self.phase_time > 15.0:
                self._begin_recover("could not reach the staging pose")

        elif self.phase == "descend_side":
            # Standing off the block by approach_offset, so this is clear of it.
            self._set_fingers_open()
            pos_error, tilt = self._track(pre_grasp, self.speed_slow, dt)
            if self._reached(pos_error, tilt, self.pos_tol_close):
                self._set_phase("insert")
            elif self.phase_time > 10.0:
                self._begin_recover("could not drop to the staging pose")

        elif self.phase == "insert":
            # Slide straight in so the block enters the open jaw.
            self._set_fingers_open()
            pos_error, tilt = self._track(grasp_pose, self.speed_slow, dt)
            if self._reached(pos_error, tilt, self.pos_tol_close):
                self._set_phase("grasp")
            elif self.phase_time > 12.0:
                self._begin_recover("could not slide onto the block")

        elif self.phase == "grasp":
            self._set_fingers_closed()
            pos_error, tilt = self._track(grasp_pose, self.speed_slow, dt)
            if self._grip_is_firm():
                self.dwell += dt
                if self.dwell > 0.3:
                    self._set_phase("squeeze")
            else:
                self.dwell = 0.0
                if self.phase_time > 4.0:
                    self._begin_recover("fingers closed but found no grip")

        elif self.phase == "squeeze":
            # Let the contacts settle before any arm motion loads them.
            self._set_fingers_closed()
            pos_error, tilt = self._hold(dt)
            if self._grip_lost(dt):
                self._begin_recover("grip slipped while settling")
            elif self.phase_time > 0.5:
                self.holding = True
                self.pick_pos = cube.copy()
                self._measure_grip_offset()
                self._set_phase("lift_check")

        elif self.phase == "lift_check":
            self._set_fingers_closed()
            pos_error, tilt = self._track(lift_pose, self.speed_slow, dt)
            if self._grip_lost(dt):
                self._begin_recover("cube slipped during the lift")
            elif self._reached(pos_error, tilt, self.pos_tol_approach):
                self.dwell += dt
                if self.dwell > 0.3:
                    self._set_phase("move_above_target")
            elif self.phase_time > 10.0:
                self._set_phase("move_above_target")

        elif self.phase == "move_above_target":
            # Stay at lift height and slide until the block is on the pad axis.
            # Once nearby, switch to a live block-XY correction so a small
            # error in the stored grip offset does not leave the object beside
            # the painted centre.
            self._set_fingers_closed()
            centering = self._block_centering_error(pad)
            if centering < 0.04:
                pos_error, tilt = self._track_place(
                    hover_block, self.speed_fast, 0.04, dt
                )
            else:
                hover_site = self._site_goal_from_offset(hover_block)
                pos_error, tilt = self._track(hover_site, self.speed_fast, dt)
            centering = self._block_centering_error(pad)
            if self._grip_lost(dt):
                self._begin_recover("lost the block in transit")
            elif centering < self.center_tol and abs(cube[2] - hold_z) < 0.04:
                self.dwell += dt
                if self.dwell > 0.4:
                    print(
                        f"[Task1Hand] centred over pad "
                        f"(off by {centering * 1000:.0f} mm), lowering"
                    )
                    self._set_phase("lower")
            else:
                self.dwell = 0.0
                if self.phase_time > 12.0 and centering < 0.020:
                    print(
                        f"[Task1Hand] close enough "
                        f"({centering * 1000:.0f} mm), lowering"
                    )
                    self._set_phase("lower")
                elif self.phase_time > 25.0:
                    self._begin_recover(
                        f"could not centre the block over the pad "
                        f"(off by {centering * 1000:.0f} mm)"
                    )

        elif self.phase == "lower":
            # Descend only after XY is on the pad axis. If the block drifts,
            # freeze Z and steer back before continuing down.
            self._set_fingers_closed()
            centering = self._block_centering_error(pad)
            if self.place_z is None:
                self.place_z = float(cube[2])
            if centering > 2.0 * self.center_tol:
                target = np.array([pad[0], pad[1], self.place_z])
                z_speed = 0.0
            else:
                floor_z = place_block[2]
                self.place_z = max(floor_z, self.place_z - self.lower_speed * dt)
                target = np.array([pad[0], pad[1], self.place_z])
                z_speed = self.lower_speed
            pos_error, tilt = self._track_place(
                target, self.speed_slow, z_speed, dt, live=False
            )
            centering = self._block_centering_error(pad)
            if self._grip_lost(dt):
                self._begin_recover("dropped the block while lowering")
            elif (
                centering < 0.020
                and cube[2] <= place_block[2] + 0.015
            ):
                self.dwell += dt
                if self.dwell > 0.3:
                    self._set_phase("release")
            else:
                self.dwell = 0.0
                if (
                    self.phase_time > 12.0
                    and cube[2] <= place_block[2] + 0.04
                    and centering < 0.030
                ):
                    print(
                        f"[Task1Hand] seating at {centering * 1000:.0f} mm "
                        "offset, opening fingers"
                    )
                    self._set_phase("release")
                elif self.phase_time > 20.0:
                    self._begin_recover(
                        f"could not lower onto the pad centre "
                        f"(off by {centering * 1000:.0f} mm)"
                    )

        elif self.phase == "release":
            # Open all seven Dex3 joints and hold the arm still so the block
            # seats on the pad instead of being dragged sideways.
            self._set_fingers_open()
            pos_error, tilt = self._hold(dt)
            if self.phase_time > 0.8 and self._fingers_open_error() < 0.15:
                self.holding = False
                self._set_phase("verify_place")
            elif self.phase_time > 3.0:
                self.holding = False
                self._set_phase("verify_place")

        elif self.phase == "verify_place":
            self._set_fingers_open()
            pos_error, tilt = self._track(retreat_pose, self.speed_slow, dt)
            self.settle_time += dt
            if self.settle_time > 1.0:
                if self._cube_on_pad():
                    self._set_phase("done")
                else:
                    self._begin_recover("cube did not settle on the pad")

        elif self.phase == "recover":
            self._set_fingers_open()
            pos_error, tilt = self._track(
                self._backed_off(grasp_pose, self.retreat_height),
                self.speed_fast,
                dt,
            )
            if self.attempts >= self.max_attempts:
                print(
                    f"[Task1Hand] giving up after {self.attempts} attempts"
                )
                self._set_phase("done")
            elif self._cube_off_table() or self._cube_tipped():
                if self._cube_speed() < 0.05:
                    self._respawn_cube()
            elif self.phase_time > 1.5:
                self._set_phase("approach_side")

        else:
            self._set_fingers_open()
            pos_error, tilt = self._hold(dt)

        if self.print_timer > self.log_period:
            self.print_timer = 0.0
            thumb, fingers, force = self._grip_state()
            site_pos, _ = self._site_pose()
            ik_residual = np.linalg.norm(
                self.goal - self._site_pos_for(self.q_ik)
            )
            joint_error = np.abs(
                self.q_ik - self.data.qpos[self.ik_qadr]
            ).max()
            print(
                f"[Task1Hand] phase={self.phase:17s} "
                f"pos_err={pos_error:.3f} tilt={tilt:4.1f} "
                f"ik_res={ik_residual:.3f} q_err={joint_error:.3f} "
                f"waist={self.data.qpos[self.ik_qadr[0]]:+.2f} "
                f"center={self._block_centering_error(pad)*1000:4.0f}mm "
                f"contacts(thumb,finger)=({thumb},{fingers}) "
                f"grip_force={force:6.2f} "
                f"site={np.round(site_pos, 3)} cube={np.round(cube, 3)}"
            )

        self._apply_pd(closing_fingers=closing)
