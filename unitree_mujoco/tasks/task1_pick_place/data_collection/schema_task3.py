"""Constants, paraphrases, and reward-term names for Task 3 collection."""

from .schema import ACTION_CLIP, ACTION_JOINTS, CONTROL_DT, IMAGE_SIZE, RECORD_HZ, RECORD_STRIDE, REWARD_CLIP

TASK_ID = "task3"
CANONICAL_INSTRUCTION = (
    "Pick up the stick, push the cube onto the green pad, and plant the stick on the cube."
)

EPISODE_KINDS = (
    "success",
    "swapped",
    "too_far",
    "no_cube",
    "space_constraint",
    "cube_too_small",
    "stick_too_short",
    "stick_lying",
)
# swapped:Stick/cube configuration is swapped.
# too_far:Required object geometry exceeds the valid manipulation range.
# no_cube:Cube is unavailable.
# space_constraint:Manipulation is blocked by workspace constraints.
# cube_too_small:Cube dimensions violate the supported range.
# stick_too_short:Stick dimensions violate the supported range.
# stick_lying:Stick begins in an invalid lying configuration.

FAILURE_REASONS = (
    "swapped",
    "too_far",
    "no_cube",
    "space_constraint",
    "cube_too_small",
    "stick_too_short",
    "stick_lying",
)

REWARD_TERMS = (
    "approach_correct",
    "grasp_correct",
    "lift_clear",
    "rotate_in_air",
    "face_contact",
    "push_toward_pad",
    "place_cube",
    "plant_stick",
    "withdraw_correct",
    "safe_abort",
    "approach_wrong",
    "grasp_wrong",
    "hand_hit_cube",
    "knock",
    "drop",
    "illegal_motion",
    "dense_toward",
    "dense_away",
)

STICK_NAMES = ("stick", "tool", "rod", "wooden stick")
CUBE_NAMES = ("cube", "block", "box", "square block")

SUCCESS_TEMPLATES = (
    "Pick up the {stick}, push the {cube} onto the green pad, and plant the stick on the cube.",
    "Grasp the {stick}, slide the {cube} onto the green square, then stand the stick on it.",
    "Use the {stick} to push the {cube} onto the green pad and leave the stick standing on the cube.",
    "Take the {stick}, drive the {cube} onto the green target, and plant the tool on top.",
    "Pick the {stick} up, push the {cube} to the green region, and set the stick on the cube.",
)

# (kind, layout_hint) — 5 success / 5 fail
SMOKE_SPECS = (
    ("success", "standard"),
    ("success", "lightness"),
    ("success", "texture"),
    ("success", "place"),
    ("success", "sized"),
    ("swapped", "swap_cube_pad"),
    ("too_far", "far_stick"),
    ("space_constraint", "blocked_approach"),
    ("cube_too_small", None),
    ("stick_too_short", None),
)


def instruction_for(kind, rng):
    stick = STICK_NAMES[int(rng.integers(0, len(STICK_NAMES)))]
    cube = CUBE_NAMES[int(rng.integers(0, len(CUBE_NAMES)))]
    template = SUCCESS_TEMPLATES[int(rng.integers(0, len(SUCCESS_TEMPLATES)))]
    text = template.format(stick=stick, cube=cube)
    if kind == "too_far":
        text = text.rstrip(".") + " (unavailable due to posture)."
    elif kind == "stick_lying":
        text = text.rstrip(".") + " (the stick is lying on the table)."
    elif kind == "no_cube":
        text = text.rstrip(".") + " (the cube is missing)."
    return text, f"{kind}:{stick}:{cube}"
