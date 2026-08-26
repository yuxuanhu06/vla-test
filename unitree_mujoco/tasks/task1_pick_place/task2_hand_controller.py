import random
import numpy as np
import mujoco


class Task2HandController:
    """Identify a requested object type, then pick and place it like Task 1.

    The run is split in two: first the controller is given the scene and the
    type instruction and must bind a single body; only after that lock does it
    run Task 1's travel / pre-grasp / grasp / lift / above-target / place
    sequence on that body.
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
    # Same as Task 1. A more abducted roll keeps the upper arm off the torso.
    REST_POSE = np.array([0.0, 0.30, 0.40, 0.0, 0.60, 0.0, -1.10, 0.0])

    # waist_yaw displaces the palm further per radian than any arm joint, so an
    # unweighted solve spends its motion there and the torso twists instead of
    # the arm extending.
    JOINT_WEIGHTS = np.array([6.0, 1.0, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5])

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

    OBJECT_TYPES = {
        "tri_prism": ("obj_tri_prism", "obj_tri_prism_geom"),
        "cylinder": ("obj_cylinder", "obj_cylinder_geom"),
        "cube": ("obj_cube", "obj_cube_geom"),
        "rect_prism": ("obj_rect_prism", "obj_rect_prism_geom"),
    }

    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.ik_data = mujoco.MjData(model)

        self.site_id = model.site("left_grasp_site").id
        self.target_id = model.body("green_target").id
        pad_geom_id = model.geom("green_target_geom").id
        self.pad_half = float(model.geom_size[pad_geom_id][0])
        self.pad_top = float(
            data.xpos[self.target_id][2] + model.geom_size[pad_geom_id][2]
        )

        # Bound only after the identify phase matches the instruction.
        self.required_type = None
        self.catalog = []
        self.cube_id = None
        self.cube_geom_id = None
        self.cube_half = 0.055
        self.cube_qadr = None
        self.cube_vadr = None
        self.cube_spawn = None

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

        # Waist may yaw a little so the left arm can reach the right-side pad,
        # but stays capped so the torso does not spin.
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
        self.max_joint_rate = 1.8

        self.kp_arm = 120.0
        self.kd_arm = 8.0
        self.kp_hold = 150.0
        self.kd_hold = 10.0
        self.kp_finger = 3.0
        self.kd_finger = 0.25
        self.kp_finger_close = 12.0
        self.kd_finger_close = 0.35

        # The jaw closes on the block's middle. The hand stages one approach
        # offset back along -APPROACH_AXIS, then slides the green palm ball in.
        self.approach_offset = 0.12
        self.grasp_dz = 0.01
        self.grasp_inset = 0.02
        self.site_radius = 0.012
        self.lift_height = 0.10
        self.retreat_height = 0.14
        self.carry_margin = 0.22
        self.carry_z = self.table_top + 2.0 * self.cube_half + self.carry_margin

        # Heights of the block itself above the pad, not of the grasp site.
        self.hover_clear = 0.06
        self.place_gap = 0.006

        # Nominal seat of the block in the jaw, refined once the grip is firm.
        self.grip_offset = np.array([self.grasp_inset, 0.0, -self.grasp_dz])

        # Every lateral move happens at this height, same as Task 1.
        self.clear_z = self.table_top + 2.0 * self.cube_half + 0.075

        self.speed_fast = 0.16
        self.speed_slow = 0.05
        self.lower_speed = 0.04

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
        self.lift_xy = None
        self.lift_wrist = None
        self.other_geom_ids = set()

        self.q_ik = self.q_home[self.ik_acts].copy()
        self._start_at_level_pose()
        self.goal = self._site_pos_for(self.q_ik)
        self.setpoint_done = True
        pad = self._pad_pos()
        print(
            f"[Task2Hand] place target (green pad) at "
            f"[{pad[0]:.3f} {pad[1]:.3f} {pad[2]:.3f}]"
        )

        self.phase = "receive"
        self.phase_time = 0.0
        self.dwell = 0.0
        self.lost_grip_time = 0.0
        self.settle_time = 0.0
        self.attempts = 0
        self.print_timer = 0.0
        self.log_period = 1.0
        self.pick_pos = None
        self.holding = False

        self._set_fingers_open()

    # ------------------------------------------------------------------
    # model helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vertical_half(model, geom_id):
        """Half-height of a geom along world-up when it is standing upright."""
        geom_type = int(model.geom_type[geom_id])
        size = model.geom_size[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return float(size[1])
        half = float(size[2])
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH and half == 0.0:
            return 0.055
        return half

    def _approach_half(self):
        """Half-size of the bound object along the side-grasp approach (world x)."""
        size = self.model.geom_size[self.cube_geom_id]
        geom_type = int(self.model.geom_type[self.cube_geom_id])
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return float(size[0])
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            half = float(size[0])
            return 0.011 if half == 0.0 else half
        return float(size[0])

    def _green_ball_on_object(self):
        """True when the green palm site is on the object, not just nearby."""
        site = self.data.site_xpos[self.site_id]
        obj = self.data.xpos[self.cube_id]
        return float(np.linalg.norm(site - obj)) < (
            self._approach_half() + self.site_radius + 0.01
        )

    def _index_middle_hit_object(self):
        """True if an open index/middle pad is already pushing the object."""
        _, fingers, _ = self._grip_state()
        return fingers >= 1 and not self._green_ball_on_object()

    def _read_scene(self):
        """All table objects the controller is allowed to know about."""
        catalog = []
        for typ, (body_name, geom_name) in self.OBJECT_TYPES.items():
            body_id = self.model.body(body_name).id
            geom_id = self.model.geom(geom_name).id
            geom_type = mujoco.mjtGeom(int(self.model.geom_type[geom_id])).name
            catalog.append(
                {
                    "type": typ,
                    "body": body_name,
                    "geom": geom_name,
                    "geom_type": geom_type.replace("mjGEOM_", "").lower(),
                    "pos": self.data.xpos[body_id].copy(),
                }
            )
        return catalog

    def _bind_target(self, item):
        """Lock pick-place onto one catalog entry. Motion starts after this."""
        self.cube_id = self.model.body(item["body"]).id
        self.cube_geom_id = self.model.geom(item["geom"]).id
        self.cube_half = self._vertical_half(self.model, self.cube_geom_id)
        self.clear_z = self.table_top + 2.0 * self.cube_half + 0.075
        self.carry_z = self.table_top + 2.0 * self.cube_half + self.carry_margin
        self.grasp_inset = 0.02
        self.grip_offset = np.array([self.grasp_inset, 0.0, -self.grasp_dz])
        cube_joint = self.model.body_jntadr[self.cube_id]
        self.cube_qadr = int(self.model.jnt_qposadr[cube_joint])
        self.cube_vadr = int(self.model.jnt_dofadr[cube_joint])
        self.cube_spawn = self.data.qpos[
            self.cube_qadr:self.cube_qadr + 7
        ].copy()
        self.pick_pos = self.data.xpos[self.cube_id].copy()
        self.other_geom_ids = set()
        for typ, (_body, geom_name) in self.OBJECT_TYPES.items():
            if typ == self.required_type:
                continue
            self.other_geom_ids.add(int(self.model.geom(geom_name).id))

    def _identify_target(self):
        """Pick the unique catalog body whose type matches the instruction."""
        matches = [item for item in self.catalog if item["type"] == self.required_type]
        if len(matches) != 1:
            raise RuntimeError(
                f"[Task2Hand] expected one '{self.required_type}', "
                f"found {len(matches)}"
            )
        return matches[0]

    def _pad_pos(self):
        """Live world pose of the green pad. Never cache Task 1's location."""
        return self.data.xpos[self.target_id].copy()

    def _apply_ik_pose(self, goal):
        """Snap the left arm to a solved IK pose. Returns False if unreachable."""
        q, residual, _ = self._solve_ik(goal)
        if residual > 0.03:
            print(
                f"[Task2Hand] IK pose off by {residual:.3f} m, "
                "keeping the current arm pose"
            )
            return False
        self.q_ik = q
        self.data.qpos[self.ik_qadr] = q
        self.data.qvel[self.ik_dofs] = 0.0
        for act_id, value in zip(self.ik_acts, q):
            self.q_target[act_id] = value
        mujoco.mj_forward(self.model, self.data)
        self.goal = self._site_pos_for(self.q_ik)
        return True

    def _seed_ik(self, goal):
        """Set q_ik to a rest-seeded solve of goal. The arm slews; qpos is not snapped."""
        q, residual, _ = self._solve_ik(goal)
        if residual > 0.03:
            return False
        self.q_ik = q
        self.goal = np.asarray(goal, float).copy()
        return True

    def _start_at_level_pose(self):
        """Same first-frame pose as Task 1: level travel over the cube XY.

        Task 1 immediately IK-solves the travel waypoint of red_cube at
        (0.34, 0.26). Reproducing that site here so receive does not hold the
        slanted START_POSE or a folded-into-torso solve at mid-table.
        """
        block = np.array([0.34, 0.26, self.table_top + 0.055])
        grasp = np.array([block[0] - 0.02, block[1], block[2] + 0.01])
        pre_grasp = np.array([grasp[0] - 0.12, grasp[1], grasp[2]])
        clear_z = self.table_top + 2.0 * 0.055 + 0.075
        self._apply_ik_pose(np.array([pre_grasp[0], pre_grasp[1], clear_z]))

    def _start_at_travel_pose(self):
        """Put the left arm on the travel waypoint of the identified object."""
        block = self.data.xpos[self.cube_id].copy()
        pre_grasp = self._backed_off(
            self._grasp_point(block), self.approach_offset
        )
        self._apply_ik_pose(self._at_clearance(pre_grasp))

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
        pad = self._pad_pos()
        print(
            f"[Task2Hand] place target (green pad) at "
            f"[{pad[0]:.3f} {pad[1]:.3f} {pad[2]:.3f}]"
        )
        grasp = self._grasp_point(cube)
        pre_grasp = self._backed_off(grasp, self.approach_offset)
        # The block hangs off the site by grip_offset, so the placing waypoints
        # are the block targets shifted back into site coordinates.
        shift = self.grip_offset * np.array([1.0, 1.0, 1.0])
        waypoints = (
            ("travel", self._at_clearance(pre_grasp)),
            ("pre_grasp", pre_grasp),
            ("grasp", grasp),
            ("lift", self._at_carry(grasp)),
            (
                "above_target",
                np.array([pad[0], pad[1], self.carry_z]) - shift,
            ),
            ("place", self._block_over_pad(pad, self.place_gap) - shift),
        )

        print(
            f"[Task2Hand] waypoint feasibility for '{self.required_type}' "
            "(travel, pre_grasp, grasp, lift, above_target, place):"
        )
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
        """Same horizontal spot, raised to the safe pre-grasp travel height."""
        return np.array([point[0], point[1], self.clear_z])

    def _at_carry(self, point):
        """Same horizontal spot at the preferred carry height."""
        return np.array([point[0], point[1], self.carry_z])

    def _neighbor_floor_z(self):
        """Lowest object height that still clears neighbor tops and hanging fingers."""
        return self.table_top + 2.0 * self.cube_half + 0.08

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
                fresh, fresh_residual, _ = self._solve_ik(
                    self.goal, seed=self.q_ik if self.holding else None, iters=400
                )
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
                fresh, fresh_residual, _ = self._solve_ik(
                    self.goal, seed=self.q_ik if self.holding else None, iters=400
                )
                if fresh_residual < residual:
                    print(
                        f"[Task2Hand] IK stalled at {residual:.3f} m, "
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

    def _reached(self, pos_error, tilt, tol):
        return self.setpoint_done and pos_error < tol and tilt < self.tilt_tol

    # ------------------------------------------------------------------
    # gripper
    # ------------------------------------------------------------------

    def _set_fingers_open(self):
        """Drive every Dex3 joint (thumb 3 + index 2 + middle 2) to open."""
        for act_id in self.finger_acts:
            self.q_target[act_id] = self.finger_open_target[act_id]

    def _set_fingers_open_blend(self, alpha):
        """Interpolate the seven Dex3 joints from the closed wrap toward open."""
        alpha = float(np.clip(alpha, 0.0, 1.0))
        for act_id in self.finger_acts:
            opened = self.finger_open_target[act_id]
            closed = self.finger_close_target.get(act_id, opened)
            self.q_target[act_id] = closed + alpha * (opened - closed)

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

    def _object_in_jaw(self):
        """True while the bound object is still sitting in the closed hand."""
        site = self._site_pose()[0]
        block = self.data.xpos[self.cube_id]
        if np.linalg.norm(block - site) > 0.08:
            return False
        thumb, fingers, force = self._grip_state()
        return thumb >= 1 or fingers >= 1 or force >= 0.3

    def _grip_lost(self, dt):
        """True once the grip has been absent for longer than the debounce."""
        held = self._object_in_jaw() if self.holding else self._grip_is_firm()
        if held:
            self.lost_grip_time = 0.0
            return False
        self.lost_grip_time += dt
        return self.lost_grip_time > self.grip_lost_debounce

    def _hit_other_object(self):
        """True if the hand or held object is pushing a table neighbor."""
        if not self.other_geom_ids:
            return False
        hand = self.thumb_geoms | self.finger_geoms
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & self.other_geom_ids and (
                self.cube_geom_id in pair or pair & hand
            ):
                return True
        return False

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
        """Block counts as placed if it is on the pad, not the painted centre.

        After a Z-only put-down the cube may sit several centimetres off
        centre; that is still a success as long as it is inside the pad.
        """
        cube = self.data.xpos[self.cube_id]
        pad = self._pad_pos()
        in_xy = float(np.linalg.norm(cube[:2] - pad[:2])) < self.pad_half - 0.02
        resting = (
            self.pad_top - 0.02 < cube[2] < self.pad_top + self.cube_half + 0.04
        )
        return bool(in_xy and resting and self._cube_speed() < 0.08)

    def _respawn_cube(self):
        self.data.qpos[self.cube_qadr:self.cube_qadr + 7] = self.cube_spawn
        self.data.qvel[self.cube_vadr:self.cube_vadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        print("[Task2Hand] cube left the table, respawned at its start pose")

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
            print(f"[Task2Hand] {self.phase} -> {phase}")
        self.phase = phase
        self.phase_time = 0.0
        self.dwell = 0.0
        self.lost_grip_time = 0.0
        self.settle_time = 0.0
        if phase == "lift_check":
            self.lift_xy = self.data.xpos[self.cube_id][:2].copy()
            self.transport_z = self.carry_z
        elif phase == "move_above_target":
            self.block_goal = self.data.xpos[self.cube_id].copy()
            self.transport_z = self.carry_z
            self.place_seeded = False
        elif phase == "lower":
            self.place_z = float(self.data.xpos[self.cube_id][2])
            self._measure_grip_offset()

    def _begin_recover(self, reason):
        print(f"[Task2Hand] {reason}; recovering (attempt {self.attempts + 1})")
        self.attempts += 1
        self.holding = False
        self._set_fingers_open()
        self._set_phase("recover")

    def step(self):
        dt = self.model.opt.timestep
        self.phase_time += dt
        self.print_timer += dt

        # ----------------------------------------------------------
        # Perceive, then match type, then start Task 1 pick-place.
        # The arm holds still until a single body is bound.
        # ----------------------------------------------------------
        if self.phase in ("receive", "understand", "identify"):
            self._set_fingers_open()
            pos_error, tilt = self._hold(dt)

            if self.phase == "receive" and self.phase_time > 0.3:
                self.catalog = self._read_scene()
                print("[Task2Hand] received scene:")
                for item in self.catalog:
                    print(
                        f"  type={item['type']:11s} body={item['body']:16s} "
                        f"geom={item['geom_type']:8s} "
                        f"pos={np.round(item['pos'], 3)}"
                    )
                self._set_phase("understand")

            elif self.phase == "understand" and self.phase_time > 0.3:
                self.required_type = random.choice(list(self.OBJECT_TYPES))
                print(
                    f"[Task2Hand] instruction: pick the '{self.required_type}' "
                    "and place it on the green pad"
                )
                self._set_phase("identify")

            elif self.phase == "identify" and self.phase_time > 0.3:
                target = self._identify_target()
                self._bind_target(target)
                print(
                    f"[Task2Hand] FINAL DECISION: pick '{target['type']}'  "
                    f"body={target['body']}  pos={np.round(target['pos'], 3)}"
                )
                print(
                    "[Task2Hand] starting Task 1 pick-place on that body only"
                )
                self._report_feasibility()
                self._set_phase("approach_side")

            self._apply_pd(closing_fingers=False)
            return

        cube = self.data.xpos[self.cube_id].copy()
        pad = self._pad_pos()
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
        lift_xy = self.lift_xy if self.lift_xy is not None else cube[:2]
        lift_block = np.array([lift_xy[0], lift_xy[1], self.carry_z])
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
            # travel: move at clearance to the pre-grasp XY of the identified object
            self._set_fingers_open()
            pos_error, tilt = self._track(travel_pose, self.speed_fast, dt)
            if self._reached(pos_error, tilt, self.pos_tol_approach):
                self._set_phase("descend_side")
            elif self.phase_time > 15.0:
                self._begin_recover("could not reach the staging pose")

        elif self.phase == "descend_side":
            # pre_grasp: drop to the side of the identified object
            self._set_fingers_open()
            pos_error, tilt = self._track(pre_grasp, self.speed_slow, dt)
            if self._reached(pos_error, tilt, self.pos_tol_close):
                self._set_phase("insert")
            elif self.phase_time > 18.0:
                self._begin_recover("could not drop to the staging pose")

        elif self.phase == "insert":
            # grasp approach: slide the open jaw onto the identified object
            self._set_fingers_open()
            if self._index_middle_hit_object():
                self._begin_recover(
                    "index/middle touched the object before the green palm ball"
                )
            else:
                pos_error, tilt = self._track(grasp_pose, self.speed_slow, dt)
                if self._reached(pos_error, tilt, self.pos_tol_close):
                    print(
                        "[Task2Hand] object at green grasp site, closing fingers"
                    )
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
            elif self.phase_time > 0.8:
                self.holding = True
                self.pick_pos = cube.copy()
                self._measure_grip_offset()
                self._set_phase("lift_check")

        elif self.phase == "lift_check":
            # Raise straight up to carry height so hanging fingers clear neighbors.
            self._set_fingers_closed()
            pos_error, tilt = self._track_place(
                lift_block, 0.02, self.speed_slow, dt, live=False
            )
            if self._grip_lost(dt):
                self._begin_recover("cube slipped during the lift")
            elif self._hit_other_object():
                self._begin_recover("lift clipped another object")
            elif cube[2] >= self.carry_z - 0.02:
                self.dwell += dt
                if self.dwell > 0.3:
                    print("[Task2Hand] carry height reached, moving to the pad")
                    self._set_phase("move_above_target")
            elif self.phase_time > 12.0:
                if cube[2] > self.table_top + 2.0 * self.cube_half + 0.08:
                    print("[Task2Hand] lift timed out, carrying at current height")
                    self.carry_z = max(self.carry_z, float(cube[2]))
                    self.transport_z = self.carry_z
                    self._set_phase("move_above_target")
                else:
                    self._begin_recover(
                        "could not lift high enough to clear neighbors"
                    )

        elif self.phase == "move_above_target":
            self._set_fingers_closed()
            if self._grip_lost(dt):
                self._begin_recover("lost the block in transit")
            elif self._hit_other_object():
                self._begin_recover("hand or object hit a neighbor in transit")
            else:
                centering = self._block_centering_error(pad)
                if centering < 0.04:
                    pos_error, tilt = self._track_place(
                        hover_block, self.speed_fast, 0.04, dt
                    )
                else:
                    hover_site = self._site_goal_from_offset(hover_block)
                    pos_error, tilt = self._track(
                        hover_site, self.speed_fast, dt
                    )
                centering = self._block_centering_error(pad)
                if centering < self.center_tol and abs(cube[2] - hold_z) < 0.04:
                    self.dwell += dt
                    if self.dwell > 0.4:
                        print(
                            f"[Task2Hand] centred over pad "
                            f"(off by {centering * 1000:.0f} mm), lowering"
                        )
                        self._set_phase("lower")
                else:
                    self.dwell = 0.0
                    if self.phase_time > 12.0 and centering < 0.020:
                        print(
                            f"[Task2Hand] close enough "
                            f"({centering * 1000:.0f} mm), lowering"
                        )
                        self._set_phase("lower")
                    elif self.phase_time > 25.0:
                        self._begin_recover(
                            f"could not centre the block over the pad "
                            f"(off by {centering * 1000:.0f} mm)"
                        )

        elif self.phase == "lower":
            # Descend until the held object almost touches the pad, then open.
            self._set_fingers_closed()
            seat_z = place_block[2]
            if self.place_z is None:
                self.place_z = float(cube[2])
            self.place_z = max(seat_z, self.place_z - self.lower_speed * dt)
            target = np.array([pad[0], pad[1], self.place_z])
            pos_error, tilt = self._track_place(
                target, 0.04, self.lower_speed, dt, live=True
            )
            centering = self._block_centering_error(pad)
            above_pad = float(cube[2] - seat_z)
            if self._grip_lost(dt):
                self._begin_recover("dropped the block while lowering")
            elif above_pad <= 0.008 and centering < self.pad_half - 0.02:
                self.dwell += dt
                if self.dwell > 0.25:
                    print(
                        f"[Task2Hand] almost touching pad "
                        f"({above_pad * 1000:.0f} mm above, "
                        f"off by {centering * 1000:.0f} mm), opening fingers"
                    )
                    self._set_phase("release")
            else:
                self.dwell = 0.0
                if self.phase_time > 30.0:
                    if (
                        above_pad <= 0.030
                        and centering < self.pad_half - 0.02
                    ):
                        print(
                            f"[Task2Hand] seating {above_pad * 1000:.0f} mm "
                            "above pad, opening fingers"
                        )
                        self._set_phase("release")
                    else:
                        self._begin_recover(
                            f"could not lower onto the pad "
                            f"({above_pad * 1000:.0f} mm above, "
                            f"off by {centering * 1000:.0f} mm)"
                        )

        elif self.phase == "release":
            # Peel the fingers off while keeping a light downward press so the
            # object stays on the pad instead of rolling away.
            self._set_fingers_open_blend(self.phase_time / 1.2)
            seat_z = place_block[2]
            target = np.array([pad[0], pad[1], seat_z])
            pos_error, tilt = self._track_place(
                target, 0.02, 0.02, dt, live=False
            )
            if self.phase_time > 1.4:
                self.holding = False
                if self._cube_on_pad():
                    print("[Task2Hand] object seated on the pad")
                    self._set_phase("withdraw")
                else:
                    self._set_phase("verify_place")
            elif self.phase_time > 3.0:
                self.holding = False
                self._set_phase("verify_place")

        elif self.phase == "verify_place":
            self._set_fingers_open()
            # Stay put until the object settles; retreating too soon knocks it.
            if self.phase_time < 1.0:
                pos_error, tilt = self._hold(dt)
            else:
                pos_error, tilt = self._track(
                    retreat_pose, self.speed_slow, dt
                )
            self.settle_time += dt
            if self.settle_time > 1.5:
                if self._cube_on_pad():
                    self._set_phase("withdraw")
                else:
                    self._begin_recover("cube did not settle on the pad")

        elif self.phase == "withdraw":
            # Lift off the seated object and ease back a little, then freeze.
            self._set_fingers_open()
            leave_pose = self._backed_off(
                np.array([pad[0], pad[1], self.clear_z]), 0.08
            )
            pos_error, tilt = self._track(leave_pose, self.speed_slow, dt)
            if self._reached(pos_error, tilt, self.pos_tol_approach):
                self._set_phase("done")
            elif self.phase_time > 4.0:
                self._set_phase("done")

        elif self.phase == "recover":
            self._set_fingers_open()
            cube_now = self.data.xpos[self.cube_id]
            pad_now = self._pad_pos()
            on_pad_xy = (
                np.linalg.norm(cube_now[:2] - pad_now[:2])
                < self.pad_half - 0.02
            )
            near_z = (
                self.pad_top - 0.03
                < cube_now[2]
                < self.pad_top + self.cube_half + 0.08
            )
            if on_pad_xy and near_z:
                print("[Task2Hand] object already on the pad, finishing")
                self.holding = False
                self._set_phase("done")
            else:
                if self._cube_off_table() or cube_now[2] < 0.3:
                    pos_error, tilt = self._hold(dt)
                    if self._cube_speed() < 0.15 or cube_now[2] < 0.3:
                        self._respawn_cube()
                    if self.attempts >= self.max_attempts:
                        print(
                            f"[Task2Hand] giving up after {self.attempts} attempts"
                        )
                        self._set_phase("done")
                    elif self.phase_time > 1.5:
                        self._set_phase("approach_side")
                else:
                    pos_error, tilt = self._track(
                        self._backed_off(grasp_pose, self.retreat_height),
                        self.speed_fast,
                        dt,
                    )
                    if self.attempts >= self.max_attempts:
                        print(
                            f"[Task2Hand] giving up after {self.attempts} attempts"
                        )
                        self._set_phase("done")
                    elif self._cube_tipped():
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
                f"[Task2Hand] phase={self.phase:17s} "
                f"pos_err={pos_error:.3f} tilt={tilt:4.1f} "
                f"ik_res={ik_residual:.3f} q_err={joint_error:.3f} "
                f"waist={self.data.qpos[self.ik_qadr[0]]:+.2f} "
                f"center={self._block_centering_error(pad)*1000:4.0f}mm "
                f"contacts(thumb,finger)=({thumb},{fingers}) "
                f"grip_force={force:6.2f} "
                f"site={np.round(site_pos, 3)} cube={np.round(cube, 3)}"
            )

        self._apply_pd(closing_fingers=closing)
