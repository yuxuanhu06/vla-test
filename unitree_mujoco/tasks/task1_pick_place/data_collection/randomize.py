"""Layout, visual, and distractor randomization for Task 1 collection."""

import numpy as np
import mujoco


CUBE_NOMINAL = np.array([0.34, 0.26, 0.906])
PAD_NOMINAL = np.array([0.31, 0.08, 0.855])
TABLE_TOP = 0.85
CUBE_HALF_Z = 0.055
JUNK_NAMES = ("junk_0", "junk_1", "junk_2")
PARK_XY = (
    (1.70, 0.70),
    (1.70, 0.78),
    (1.78, 0.70),
)


def _set_free_pose(model, data, body_name, xyz, quat):
    body_id = model.body(body_name).id
    jnt = model.body_jntadr[body_id]
    qadr = model.jnt_qposadr[jnt]
    vadr = model.jnt_dofadr[jnt]
    data.qpos[qadr : qadr + 3] = xyz
    data.qpos[qadr + 3 : qadr + 7] = quat
    data.qvel[vadr : vadr + 6] = 0.0


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _tip_quat(rng):
    """Lie the tall block on its side so a top grasp would be required."""
    axis = rng.choice(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    half = 0.5 * (np.pi / 2.0)
    q = np.array([np.cos(half), 0.0, 0.0, 0.0])
    q[1:4] = np.sin(half) * axis
    return q


def _set_pad_xy(model, xy):
    body_id = model.body("green_target").id
    model.body_pos[body_id, 0] = float(xy[0])
    model.body_pos[body_id, 1] = float(xy[1])
    model.body_pos[body_id, 2] = PAD_NOMINAL[2]


def _clearance_ok(cube_xy, pad_xy, junk_xy):
    if cube_xy[0] - 0.12 < 0.14:
        return False
    if np.linalg.norm(cube_xy - pad_xy) < 0.12:
        return False
    for junk in junk_xy:
        if junk is None:
            continue
        # Keep the -X approach lane empty.
        if abs(junk[1] - cube_xy[1]) < 0.06 and cube_xy[0] - 0.14 < junk[0] < cube_xy[0]:
            return False
        if np.linalg.norm(junk - cube_xy) < 0.08:
            return False
        if np.linalg.norm(junk - pad_xy) < 0.12:
            return False
    return True


def _sample_success_layout(rng):
    for _ in range(40):
        cube_xy = CUBE_NOMINAL[:2] + rng.uniform(-0.04, 0.04, size=2)
        pad_xy = PAD_NOMINAL[:2] + rng.uniform(-0.04, 0.04, size=2)
        cube_xy[0] = float(np.clip(cube_xy[0], 0.28, 0.42))
        cube_xy[1] = float(np.clip(cube_xy[1], 0.16, 0.32))
        pad_xy[0] = float(np.clip(pad_xy[0], 0.24, 0.40))
        pad_xy[1] = float(np.clip(pad_xy[1], -0.02, 0.16))
        n_junk = int(rng.integers(0, 4))
        junk_xy = [None, None, None]
        for i in range(n_junk):
            jxy = np.array(
                [rng.uniform(0.48, 0.72), rng.uniform(-0.22, 0.08)]
            )
            junk_xy[i] = jxy
        if _clearance_ok(cube_xy, pad_xy, junk_xy):
            return {
                "cube_xy": cube_xy,
                "pad_xy": pad_xy,
                "cube_yaw": float(rng.uniform(-np.pi / 10.0, np.pi / 10.0)),
                "tipped": False,
                "junk_xy": junk_xy,
                "subtype": "working_band",
            }
    return {
        "cube_xy": CUBE_NOMINAL[:2].copy(),
        "pad_xy": PAD_NOMINAL[:2].copy(),
        "cube_yaw": 0.0,
        "tipped": False,
        "junk_xy": [None, None, None],
        "subtype": "nominal_fallback",
    }


def _sample_space_layout(rng):
    subtype = rng.choice(["tight_x", "blocked_approach", "pad_jammed"])
    cube_xy = CUBE_NOMINAL[:2].copy()
    pad_xy = PAD_NOMINAL[:2].copy()
    junk_xy = [None, None, None]
    if subtype == "tight_x":
        cube_xy[0] = float(rng.uniform(0.16, 0.21))
        cube_xy[1] = float(rng.uniform(0.18, 0.30))
        pad_xy[1] = float(rng.uniform(-0.02, 0.12))
    elif subtype == "blocked_approach":
        cube_xy = CUBE_NOMINAL[:2] + rng.uniform(-0.02, 0.02, size=2)
        junk_xy[0] = np.array([cube_xy[0] - 0.07, cube_xy[1]])
    else:
        cube_xy = CUBE_NOMINAL[:2] + rng.uniform(-0.02, 0.02, size=2)
        pad_xy = cube_xy + np.array([0.03, -0.02])
    return {
        "cube_xy": cube_xy,
        "pad_xy": pad_xy,
        "cube_yaw": float(rng.uniform(-np.pi / 12.0, np.pi / 12.0)),
        "tipped": False,
        "junk_xy": junk_xy,
        "subtype": subtype,
    }


def _sample_kinematics_layout(rng):
    subtype = rng.choice(["tipped", "far_y"])
    cube_xy = CUBE_NOMINAL[:2].copy()
    pad_xy = PAD_NOMINAL[:2].copy()
    tipped = False
    if subtype == "tipped":
        cube_xy = CUBE_NOMINAL[:2] + rng.uniform(-0.03, 0.03, size=2)
        tipped = True
    else:
        cube_xy[0] = float(rng.uniform(0.30, 0.40))
        cube_xy[1] = float(rng.choice([-0.36, 0.36]))
        pad_xy[1] = float(np.sign(cube_xy[1]) * 0.30)
    return {
        "cube_xy": cube_xy,
        "pad_xy": pad_xy,
        "cube_yaw": float(rng.uniform(-np.pi / 10.0, np.pi / 10.0)),
        "tipped": tipped,
        "junk_xy": [None, None, None],
        "subtype": subtype,
    }


def sample_layout(kind, rng):
    if kind == "space_constraint":
        return _sample_space_layout(rng)
    if kind == "kinematics_limit":
        return _sample_kinematics_layout(rng)
    return _sample_success_layout(rng)


def apply_visuals(model, rng):
    finish = "matte" if rng.random() < 0.5 else "smooth"
    cube_mat = model.material("cube_finish_mat").id
    table_mat = model.material("table_finish_mat").id
    floor_mat = model.material("task_groundplane_mat").id

    cube_rgb = rng.uniform(0.15, 0.95, size=3)
    # Keep one channel high so it stays a distinct "object" color.
    cube_rgb[int(rng.integers(0, 3))] = float(rng.uniform(0.7, 1.0))
    model.mat_rgba[cube_mat, :3] = cube_rgb
    model.mat_rgba[cube_mat, 3] = 1.0
    if finish == "matte":
        model.mat_reflectance[cube_mat] = 0.0
        if hasattr(model, "mat_specular"):
            model.mat_specular[cube_mat] = 0.05
    else:
        model.mat_reflectance[cube_mat] = 0.6
        if hasattr(model, "mat_specular"):
            model.mat_specular[cube_mat] = 0.7

    table_rgb = rng.uniform(0.15, 0.7, size=3)
    model.mat_rgba[table_mat, :3] = table_rgb
    model.mat_rgba[table_mat, 3] = 1.0
    model.mat_reflectance[table_mat] = 0.0 if rng.random() < 0.5 else 0.35

    model.mat_rgba[floor_mat, :3] = rng.uniform(0.1, 0.45, size=3)

    model.vis.headlight.ambient[:] = rng.uniform(0.15, 0.45, size=3)
    model.vis.headlight.diffuse[:] = rng.uniform(0.35, 0.85, size=3)
    model.vis.headlight.specular[:] = rng.uniform(0.0, 0.25, size=3)

    light_id = model.light("task_key_light").id
    model.light_pos[light_id] = np.array(
        [rng.uniform(-0.2, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(1.2, 1.9)]
    )
    model.light_dir[light_id] = np.array([0.0, 0.0, -1.0])
    model.light_diffuse[light_id] = rng.uniform(0.4, 0.9, size=3)

    haze = rng.uniform(0.05, 0.4, size=3)
    model.vis.rgba.haze[:3] = haze
    return {
        "cube_finish": finish,
        "cube_rgb": cube_rgb.tolist(),
        "table_rgb": table_rgb.tolist(),
    }


def apply_layout(model, data, layout, rng):
    if layout["tipped"]:
        # Half-width along the fallen axis is 0.055, so COM sits lower.
        cube_z = TABLE_TOP + 0.011 + 0.002
        quat = _quat_mul(_yaw_quat(layout["cube_yaw"]), _tip_quat(rng))
    else:
        cube_z = TABLE_TOP + CUBE_HALF_Z + 0.001
        quat = _yaw_quat(layout["cube_yaw"])

    _set_free_pose(
        model,
        data,
        "red_cube",
        [layout["cube_xy"][0], layout["cube_xy"][1], cube_z],
        quat,
    )
    _set_pad_xy(model, layout["pad_xy"])

    placed = []
    for i, name in enumerate(JUNK_NAMES):
        xy = layout["junk_xy"][i]
        if xy is None:
            _set_free_pose(
                model,
                data,
                name,
                [PARK_XY[i][0], PARK_XY[i][1], 0.03],
                _yaw_quat(0.0),
            )
        else:
            _set_free_pose(
                model,
                data,
                name,
                [float(xy[0]), float(xy[1]), TABLE_TOP + 0.03],
                _yaw_quat(0.0),
            )
            placed.append(
                {
                    "name": name,
                    "xy": [float(xy[0]), float(xy[1])],
                }
            )
    mujoco.mj_forward(model, data)
    return placed


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def randomize_episode(model, data, kind, rng):
    visuals = apply_visuals(model, rng)
    layout = sample_layout(kind, rng)
    junk = apply_layout(model, data, layout, rng)
    return {
        "kind": kind,
        "layout_subtype": layout["subtype"],
        "cube_xy": [float(layout["cube_xy"][0]), float(layout["cube_xy"][1])],
        "pad_xy": [float(layout["pad_xy"][0]), float(layout["pad_xy"][1])],
        "cube_yaw": float(layout["cube_yaw"]),
        "tipped": bool(layout["tipped"]),
        "distractors": junk,
        **visuals,
    }
