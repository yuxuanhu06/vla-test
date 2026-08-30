"""Constants, paraphrases, and reward-term names for Task 2 collection."""

from .schema import ACTION_CLIP, ACTION_JOINTS, CONTROL_DT, IMAGE_SIZE, RECORD_HZ, RECORD_STRIDE, REWARD_CLIP

TASK_ID = "task2"
CANONICAL_INSTRUCTION = (
    "Pick up the requested object and place it on the green pad."
)

OBJECT_TYPES = ("tri_prism", "cylinder", "cube", "rect_prism")

EPISODE_KINDS = (
    "success",
    "space_constraint",
    "kinematics_limit",
    "tipped",
    "wrong_object",
    "neighbor_collision",
)
FAILURE_REASONS = (
    "space_constraint",
    "kinematics_limit",
    "tipped",
    "wrong_object",
    "knock",
)

REWARD_TERMS = (
    "approach_correct",
    "grasp_correct",
    "lift_clear",
    "carry_toward_pad",
    "place_centered",
    "withdraw_correct",
    "safe_abort",
    "identify_correct",
    "neighbor_clear",
    "approach_wrong",
    "grasp_wrong",
    "carry_wrong",
    "knock",
    "drop",
    "illegal_motion",
    "identify_wrong",
    "select_wrong",
    "knock_neighbor",
    "wrong_place",
    "dense_toward",
    "dense_away",
)

TYPE_PHRASES = {
    "tri_prism": ("triangular prism", "triangle block", "tri prism"),
    "cylinder": ("cylinder", "round column", "tube"),
    "cube": ("cube", "square block", "box"),
    "rect_prism": ("rectangular prism", "rect prism", "rectangular block"),
}

SUCCESS_TEMPLATES = (
    "Pick up the {name} and place it on the green pad.",
    "Grasp the {name} with the left hand and set it on the green target.",
    "Take the {name} and put it onto the green square.",
    "Place the {name} on the green pad.",
    "Pick the {name} up and deliver it to the green region.",
)

# (kind, layout_hint, type_hint) — type_hint is used for success type balance.
SMOKE_SPECS = (
    ("success", "working_row", "tri_prism"),
    ("success", "arc_small", "cylinder"),
    ("success", "mixed_far", "cube"),
    ("success", "mixed_tipped", "rect_prism"),
    ("space_constraint", "too_close", None),
    ("kinematics_limit", "mixed_far", None),
    ("tipped", "mixed_tipped", None),
    ("wrong_object", "working_row", "tri_prism"),
    ("neighbor_collision", "too_close", None),
    ("space_constraint", "blocked_approach", None),
)


def instruction_for(kind, required_type, rng):
    phrases = TYPE_PHRASES[required_type]
    name = phrases[int(rng.integers(0, len(phrases)))]
    template = SUCCESS_TEMPLATES[int(rng.integers(0, len(SUCCESS_TEMPLATES)))]
    text = template.format(name=name)
    if kind == "kinematics_limit":
        text = text.rstrip(".") + " (unavailable due to posture)."
    elif kind == "tipped":
        text = text.rstrip(".") + " (object is lying on the table)."
    elif kind == "wrong_object":
        text = text.rstrip(".") + " (the requested object is not on the table)."
    return text, f"{required_type}:{kind}"
