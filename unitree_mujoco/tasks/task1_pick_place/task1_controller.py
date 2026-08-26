import numpy as np
import mujoco


class Task1Controller:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.hand_name = "left_wrist_roll_link"
        self.cube_name = "red_cube"

        self.hand_id = self.model.body(self.hand_name).id
        self.cube_id = self.model.body(self.cube_name).id

        # -------------------------------------------------
        # Save every actuated joint's initial position
        # -------------------------------------------------
        self.q_home = np.zeros(self.model.nu)

        for act_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[act_id, 0]
            qpos_idx = self.model.jnt_qposadr[joint_id]

            self.q_home[act_id] = self.data.qpos[qpos_idx]

        # Desired joint configuration.
        # This is what IK modifies over time.
        self.q_target = self.q_home.copy()

        # -------------------------------------------------
        # Find left-arm actuators + exact MuJoCo addresses
        # -------------------------------------------------
        self.left_arm_actuator_ids = []
        self.left_arm_qpos_ids = []
        self.left_arm_qvel_ids = []

        for act_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[act_id, 0]
            joint_name = self.model.joint(joint_id).name

            if (
                "left_shoulder" in joint_name
                or "left_elbow" in joint_name
                or "left_wrist" in joint_name
            ):
                self.left_arm_actuator_ids.append(act_id)

                self.left_arm_qpos_ids.append(
                    self.model.jnt_qposadr[joint_id]
                )

                self.left_arm_qvel_ids.append(
                    self.model.jnt_dofadr[joint_id]
                )

        # -------------------------------------------------
        # IK parameters
        # -------------------------------------------------
        self.ik_gain = 0.5
        self.max_ik_step = 0.01
        self.damping = 0.01

        # -------------------------------------------------
        # PD gains
        # motor actuator => ctrl = torque
        # -------------------------------------------------
        self.kp_arm = 80.0
        self.kd_arm = 8.0

        self.kp_hold = 80.0
        self.kd_hold = 8.0

        # Stop 5 cm before cube
        self.stop_distance = 0.05


    def step(self):

        # =================================================
        # 1. Current hand / cube positions
        # =================================================
        hand_pos = self.data.xpos[self.hand_id].copy()
        cube_pos = self.data.xpos[self.cube_id].copy()

        direction = cube_pos - hand_pos
        distance = np.linalg.norm(direction)

        print(
            f"Hand Position: {hand_pos}, "
            f"Cube Position: {cube_pos}, "
            f"Distance: {distance:.4f}"
        )

        # =================================================
        # 2. Only run IK while farther than 5 cm
        # =================================================
        if distance > self.stop_distance:

            direction_unit = direction / distance

            # Do not aim for cube center.
            # Aim for a point 5 cm away from cube.
            target_pos = (
                cube_pos
                - direction_unit * self.stop_distance
            )

            pos_error = target_pos - hand_pos

            # ---------------------------------------------
            # Full translational Jacobian
            # ---------------------------------------------
            jac = np.zeros((3, self.model.nv))

            mujoco.mj_jacBody(
                self.model,
                self.data,
                jac,
                None,
                self.hand_id,
            )

            # ---------------------------------------------
            # Extract ONLY left arm Jacobian columns
            # shape = 3 x 7
            # ---------------------------------------------
            J_arm = jac[:, self.left_arm_qvel_ids]

            # ---------------------------------------------
            # Damped Least Squares
            #
            # dq = J^T (J J^T + lambda I)^-1 e
            # ---------------------------------------------
            A = (
                J_arm @ J_arm.T
                + self.damping * np.eye(3)
            )

            dq_arm = (J_arm.T@ np.linalg.solve(A, pos_error))

            dq_arm *= self.ik_gain

            # Prevent giant IK jumps
            dq_norm = np.linalg.norm(dq_arm)

            if dq_norm > self.max_ik_step:
                dq_arm *= (self.max_ik_step / dq_norm)

            # ---------------------------------------------
            # Integrate IK into persistent q_target
            # ---------------------------------------------
            for local_id, act_id in enumerate(self.left_arm_actuator_ids):
                joint_id = self.model.actuator_trnid[
                    act_id, 0
                ]

                joint_range = self.model.jnt_range[joint_id]

                self.q_target[act_id] += dq_arm[local_id]

                self.q_target[act_id] = np.clip(self.q_target[act_id],joint_range[0],joint_range[1],)

        # =================================================
        # 3. PD CONTROL
        #
        # XML uses <motor>, so ctrl must be TORQUE.
        # =================================================
        for act_id in range(self.model.nu):

            joint_id = self.model.actuator_trnid[
                act_id, 0
            ]

            qpos_idx = self.model.jnt_qposadr[
                joint_id
            ]

            qvel_idx = self.model.jnt_dofadr[
                joint_id
            ]

            q = self.data.qpos[qpos_idx]
            dq = self.data.qvel[qvel_idx]

            # ---------------------------------------------
            # Left arm follows IK target
            # ---------------------------------------------
            if act_id in self.left_arm_actuator_ids:

                torque = (
                    self.kp_arm
                    * (self.q_target[act_id] - q)
                    - self.kd_arm * dq
                )

            # ---------------------------------------------
            # Everything else holds initial pose
            # ---------------------------------------------
            else:
                torque = (
                    self.kp_hold
                    * (self.q_home[act_id] - q)
                    - self.kd_hold * dq
                )

            # ---------------------------------------------
            # Respect actuator ctrlrange
            # ---------------------------------------------
            if self.model.actuator_ctrllimited[act_id]:

                ctrl_min = self.model.actuator_ctrlrange[act_id, 0]

                ctrl_max = self.model.actuator_ctrlrange[act_id, 1]

                torque = np.clip(torque,ctrl_min, ctrl_max,)

            self.data.ctrl[act_id] = torque