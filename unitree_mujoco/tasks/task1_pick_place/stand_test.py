import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "tasks/task1_pick_place/task1.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

# 保存 G1 初始姿态
initial_qpos = data.qpos.copy()

# 29 个 actuator 对应的 joint
actuator_joint_ids = []

for aid in range(model.nu):
    jid = model.actuator_trnid[aid, 0]
    actuator_joint_ids.append(jid)

actuator_joint_ids = np.array(actuator_joint_ids)

qpos_ids = np.array([
    model.jnt_qposadr[jid]
    for jid in actuator_joint_ids
])

dof_ids = np.array([
    model.jnt_dofadr[jid]
    for jid in actuator_joint_ids
])

q_des = initial_qpos[qpos_ids].copy()

# Left leg
q_des[0] = -0.1
q_des[1] = 0.0
q_des[2] = 0.0
q_des[3] = 0.3
q_des[4] = -0.2
q_des[5] = 0.0

# Right leg
q_des[6] = -0.1
q_des[7] = 0.0
q_des[8] = 0.0
q_des[9] = 0.3
q_des[10] = -0.2
q_des[11] = 0.0

# Waist
q_des[12] = 0.0
q_des[13] = 0.0
q_des[14] = 0.0

# Arms
q_des[15:29] = 0.0

# 一定要放在上面这些赋值之后
print("Number of actuators:", model.nu)
print("Standing target:")
print(q_des)

# 第一版统一 PD 参数
KP = np.ones(model.nu) * 40.0
KD = np.ones(model.nu) * 2.0

# 腿部需要更硬一点
KP[0:12] = 80.0
KD[0:12] = 4.0

# 腰部
KP[12:15] = 60.0
KD[12:15] = 3.0

# 手臂
KP[15:29] = 30.0
KD[15:29] = 1.5


def standing_pd():
    q = data.qpos[qpos_ids]
    dq = data.qvel[dof_ids]

    tau = KP * (q_des - q) - KD * dq

    for aid in range(model.nu):
        if model.actuator_ctrllimited[aid]:
            lo, hi = model.actuator_ctrlrange[aid]
            data.ctrl[aid] = np.clip(tau[aid], lo, hi)
        else:
            data.ctrl[aid] = tau[aid]


with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        start = time.time()

        standing_pd()

        mujoco.mj_step(model, data)

        viewer.sync()

        dt = model.opt.timestep - (time.time() - start)

        if dt > 0:
            time.sleep(dt)
