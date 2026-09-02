"""Constants, paraphrases, and reward-term names for Task 1 collection."""

CONTROL_DT = 0.005
RECORD_HZ = 10.0
RECORD_STRIDE = int(round(1.0 / (RECORD_HZ * CONTROL_DT)))
IMAGE_SIZE = 224
ACTION_CLIP = 0.25
REWARD_CLIP = 2.0

TASK_ID = "task1"

EPISODE_KINDS = ("success", "space_constraint", "kinematics_limit")
# space_constraint: Manipulation is blocked by workspace or spacing constraints.
# kinematics_limit: Required configuration is unavailable due to robot posture or reachability.

FAILURE_REASONS = ("space_constraint", "kinematics_limit")

ACTION_JOINTS = (
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
)

REWARD_TERMS = (
    "approach_correct",
    "grasp_correct",
    "lift_clear",
    "carry_toward_pad",
    "place_centered",
    "withdraw_correct",
    "safe_abort",
    "approach_wrong",
    "grasp_wrong",
    "carry_wrong",
    "knock",
    "drop",
    "illegal_motion",
    "dense_toward",
    "dense_away",
)

SUCCESS_INSTRUCTIONS = (
    "Pick up the red block and place it on the green pad.",
    "Pick up the object and move it to the green target.",
    "Grasp the block with the left hand and set it on the green square.",
    "Take the cube and put it onto the green target.",
    "Place the red object on the green pad.",
    "Pick the block up and deliver it to the green region.",
)

SPACE_INSTRUCTIONS = SUCCESS_INSTRUCTIONS

KINEMATICS_INSTRUCTIONS = (
    "Pick up the red block (unavailable due to posture).",
    "Pick up the object (unavailable due to posture).",
    "Grasp the block with the left hand (unavailable due to posture).",
    "Take the cube and put it onto the green target (unavailable due to posture).",
)

SMOKE_KINDS = (
    "success",
    "success",
    "success",
    "success",
    "success",
    "space_constraint",
    "space_constraint",
    "space_constraint",
    "kinematics_limit",
    "kinematics_limit",
)


def instruction_for(kind, rng):
    if kind == "kinematics_limit":
        pool = KINEMATICS_INSTRUCTIONS
    elif kind == "space_constraint":
        pool = SPACE_INSTRUCTIONS
    else:
        pool = SUCCESS_INSTRUCTIONS
    index = int(rng.integers(0, len(pool)))
    return pool[index], index
