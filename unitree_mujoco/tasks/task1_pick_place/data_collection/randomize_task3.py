"""Layout, size, visual, and distractor randomization for Task 3 collection."""

import numpy as np
import mujoco


TABLE_TOP = 0.85
PAD_Z = 0.8503
STICK_RADIUS = 0.015
STICK_HALF_NOM = 0.10
CUBE_HALF_NOM = 0.04
STICK_MASS_NOM = 0.10
CUBE_MASS_NOM = 0.20
START_CELL = np.array([0.34, 0.26])
CUBE_PARK = (1.70, 0.70, 0.03)
SPHERE_PARK = (1.70, 0.86, 0.04)
JUNK_PARK = ((1.70, 0.70, 0.03), (1.78, 0.70, 0.03))
JUNK_NAMES = ("junk_0", "junk_1")
SPHERE_R = 0.028

# Expert-feasible success band (24 cm stick / 6 cm cube knock or stall).
STICK_HALF_SUCCESS = (0.100, 0.105)
CUBE_HALF_SUCCESS = (0.035, 0.042)
STICK_HALF_SHORT = 0.04
CUBE_HALF_TINY = 0.012


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


def _tip_quat(rng, yaw=0.0):
    axis = rng.choice(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    half = 0.5 * (np.pi / 2.0)
    tip = np.array([np.cos(half), 0.0, 0.0, 0.0])
    tip[1:4] = np.sin(half) * axis
    return _quat_mul(_yaw_quat(yaw), tip)


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


def _set_pad_xy(model, xy):
    body_id = model.body("green_target").id
    model.body_pos[body_id, 0] = float(xy[0])
    model.body_pos[body_id, 1] = float(xy[1])
    model.body_pos[body_id, 2] = PAD_Z


def _set_mass(model, geom_name, body_name, mass):
    gid = model.geom(geom_name).id
    bid = model.body(body_name).id
    if hasattr(model, "geom_mass"):
        model.geom_mass[gid] = float(mass)
    model.body_mass[bid] = float(mass)


def apply_sizes(model, stick_half, cube_half):
    stick_gid = model.geom("tool_stick_geom").id
    cube_gid = model.geom("obj_cube_geom").id
    model.geom_size[stick_gid, 0] = STICK_RADIUS
    model.geom_size[stick_gid, 1] = float(stick_half)
    model.geom_size[cube_gid, :3] = float(cube_half)
    stick_mass = STICK_MASS_NOM * (float(stick_half) / STICK_HALF_NOM)
    cube_mass = CUBE_MASS_NOM * (float(cube_half) / CUBE_HALF_NOM) ** 3
    _set_mass(model, "tool_stick_geom", "tool_stick", stick_mass)
    _set_mass(model, "obj_cube_geom", "obj_cube", cube_mass)


def _nominal_pose():
    return {
        "stick_xy": np.array([0.40, 0.26]),
        "cube_xy": np.array([0.40, 0.10]),
        "pad_xy": np.array([0.44, -0.12]),
        "junk_xy": [None, None],
        "stick_yaw": 0.0,
        "cube_yaw": 0.0,
    }


def _sample_success_sizes(rng, subtype):
    if subtype == "sized":
        stick_half = float(rng.uniform(0.102, 0.105))
        cube_half = float(rng.choice(np.array([0.035, 0.037, 0.038])))
    elif subtype in ("standard", "lightness", "texture", "place"):
        stick_half = STICK_HALF_NOM
        cube_half = CUBE_HALF_NOM
    elif subtype == "long_stick":
        stick_half = float(rng.uniform(0.104, 0.106))
        cube_half = float(rng.uniform(0.035, 0.040))
    else:
        stick_half = float(rng.uniform(*STICK_HALF_SUCCESS))
        cube_half = float(rng.uniform(*CUBE_HALF_SUCCESS))
    return stick_half, cube_half


def _lane_clear(stick_xy, cube_xy, pad_xy, junk_xy, stick_half):
    if stick_xy[0] < 0.37 or stick_xy[0] - 0.12 < 0.16:
        return False
    if abs(stick_xy[0] - START_CELL[0]) < 0.03 and abs(stick_xy[1] - START_CELL[1]) < 0.04:
        return False
    min_gap = max(0.12, stick_half + 0.02)
    if stick_xy[1] - cube_xy[1] < min_gap:
        return False
    if cube_xy[1] - pad_xy[1] < 0.14:
        return False
    if abs(cube_xy[0] - stick_xy[0]) > 0.02:
        return False
    if abs(pad_xy[0] - cube_xy[0]) > 0.08:
        return False
    if np.linalg.norm(cube_xy - START_CELL) < 0.08:
        return False
    for junk in junk_xy:
        if junk is None:
            continue
        if abs(junk[1] - stick_xy[1]) < 0.06 and stick_xy[0] - 0.14 < junk[0] < stick_xy[0]:
            return False
        if np.linalg.norm(junk - stick_xy) < 0.10:
            return False
        if np.linalg.norm(junk - cube_xy) < 0.10:
            return False
        if np.linalg.norm(junk - pad_xy) < 0.12:
            return False
    return True


def _sample_working_xy(rng, subtype, stick_half):
    if subtype != "place":
        return _nominal_pose()
    for _ in range(60):
        stick_xy = np.array(
            [rng.uniform(0.400, 0.420), rng.uniform(0.255, 0.270)]
        )
        cube_xy = np.array(
            [
                stick_xy[0] + rng.uniform(-0.010, 0.010),
                rng.uniform(0.095, 0.110),
            ]
        )
        pad_xy = np.array(
            [rng.uniform(0.430, 0.460), rng.uniform(-0.140, -0.100)]
        )
        if _lane_clear(stick_xy, cube_xy, pad_xy, [None, None], stick_half):
            return {
                "stick_xy": stick_xy,
                "cube_xy": cube_xy,
                "pad_xy": pad_xy,
                "junk_xy": [None, None],
                "stick_yaw": float(rng.uniform(-0.08, 0.08)),
                "cube_yaw": float(rng.uniform(-0.08, 0.08)),
            }
    return _nominal_pose()


def _sample_success(rng, subtype):
    size_key = subtype[5:] if subtype.startswith("safe_") else subtype
    stick_half, cube_half = _sample_success_sizes(rng, size_key)
    if subtype.startswith("safe_"):
        pose = _nominal_pose()
        if size_key == "place":
            pose["stick_xy"] = np.array([0.410, 0.265])
            pose["cube_xy"] = np.array([0.410, 0.105])
            pose["pad_xy"] = np.array([0.450, -0.110])
    elif size_key == "place":
        pose = _sample_working_xy(rng, size_key, stick_half)
    else:
        pose = _nominal_pose()
    pose.update(
        {
            "subtype": size_key,
            "stick_half": stick_half,
            "cube_half": cube_half,
            "stick_lying": False,
            "cube_present": True,
            "block_approach": False,
        }
    )
    return pose


def _sample_swapped(rng, subtype):
    layout = _sample_success(rng, "working_band")
    if subtype == "swap_stick_pad":
        layout["stick_xy"], layout["pad_xy"] = (
            layout["pad_xy"].copy(),
            layout["stick_xy"].copy(),
        )
        layout["subtype"] = "swap_stick_pad"
    else:
        layout["cube_xy"], layout["pad_xy"] = (
            layout["pad_xy"].copy(),
            layout["cube_xy"].copy(),
        )
        layout["subtype"] = "swap_cube_pad"
    return layout


def _sample_too_far(rng, subtype):
    layout = _sample_success(rng, "working_band")
    choice = subtype or str(rng.choice(np.array(["far_stick", "far_cube", "far_pad"])))
    if choice == "far_cube":
        layout["cube_xy"] = np.array(
            [rng.uniform(0.70, 0.84), rng.choice(np.array([-0.34, 0.36]))]
        )
    elif choice == "far_pad":
        layout["pad_xy"] = np.array(
            [rng.uniform(0.30, 0.50), rng.choice(np.array([-0.36, 0.34]))]
        )
    else:
        layout["stick_xy"] = np.array(
            [rng.uniform(0.30, 0.42), rng.choice(np.array([-0.34, 0.36]))]
        )
        choice = "far_stick"
    layout["subtype"] = choice
    layout["junk_xy"] = [None, None]
    return layout


def _sample_space(rng, subtype):
    layout = _sample_success(rng, "working_band")
    choice = subtype or str(rng.choice(np.array(["tight_x", "blocked_approach"])))
    if choice == "tight_x":
        layout["stick_xy"][0] = float(rng.uniform(0.16, 0.21))
        layout["cube_xy"][0] = layout["stick_xy"][0] + float(rng.uniform(-0.01, 0.02))
        layout["block_approach"] = False
    else:
        layout["block_approach"] = True
        choice = "blocked_approach"
    layout["subtype"] = choice
    return layout


def _sample_no_cube(rng):
    layout = _sample_success(rng, "working_band")
    layout["cube_present"] = False
    layout["subtype"] = "no_cube"
    return layout


def _sample_cube_too_small(rng):
    layout = _sample_success(rng, "working_band")
    layout["cube_half"] = CUBE_HALF_TINY
    layout["subtype"] = "cube_too_small"
    return layout


def _sample_stick_too_short(rng):
    layout = _sample_success(rng, "working_band")
    layout["stick_half"] = STICK_HALF_SHORT
    layout["subtype"] = "stick_too_short"
    return layout


def _sample_stick_lying(rng):
    layout = _sample_success(rng, "working_band")
    layout["stick_lying"] = True
    layout["subtype"] = "stick_lying"
    return layout


def sample_layout(kind, rng, layout_hint=None):
    if kind == "success":
        subtype = layout_hint or str(
            rng.choice(np.array(["standard", "lightness", "texture", "place", "sized"]))
        )
        return _sample_success(rng, subtype)
    if kind == "swapped":
        return _sample_swapped(rng, layout_hint or "swap_cube_pad")
    if kind == "too_far":
        return _sample_too_far(rng, layout_hint)
    if kind == "space_constraint":
        return _sample_space(rng, layout_hint)
    if kind == "no_cube":
        return _sample_no_cube(rng)
    if kind == "cube_too_small":
        return _sample_cube_too_small(rng)
    if kind == "stick_too_short":
        return _sample_stick_too_short(rng)
    if kind == "stick_lying":
        return _sample_stick_lying(rng)
    return _sample_success(rng, "working_band")


def apply_visuals(model, rng, mode=None):
    if mode == "texture":
        table_texture = "checker"
    elif mode == "standard":
        table_texture = "png"
    else:
        table_texture = "png" if rng.random() < 0.6 else "checker"
    table_geom = model.geom("table_top").id
    png_id = model.material("table_png_mat").id
    checker_id = model.material("table_checker_mat").id
    model.geom_matid[table_geom] = png_id if table_texture == "png" else checker_id
    table_mat = png_id if table_texture == "png" else checker_id
    table_rgb = rng.uniform(0.25, 0.75, size=3)
    model.mat_rgba[table_mat, :3] = table_rgb
    model.mat_rgba[table_mat, 3] = 1.0
    table_finish = "matte" if rng.random() < 0.5 else "smooth"
    model.mat_reflectance[table_mat] = 0.0 if table_finish == "matte" else 0.35

    floor_mat = model.material("task_groundplane_mat").id
    model.mat_rgba[floor_mat, :3] = rng.uniform(0.1, 0.45, size=3)

    object_visuals = {}
    for name, mat_name in (
        ("stick", "stick_finish_mat"),
        ("cube", "cube_finish_mat"),
    ):
        mat = model.material(mat_name).id
        rgb = rng.uniform(0.15, 0.95, size=3)
        rgb[int(rng.integers(0, 3))] = float(rng.uniform(0.7, 1.0))
        finish = "matte" if rng.random() < 0.5 else "smooth"
        model.mat_rgba[mat, :3] = rgb
        model.mat_rgba[mat, 3] = 1.0
        if finish == "matte":
            model.mat_reflectance[mat] = 0.0
            if hasattr(model, "mat_specular"):
                model.mat_specular[mat] = 0.05
        else:
            model.mat_reflectance[mat] = 0.55
            if hasattr(model, "mat_specular"):
                model.mat_specular[mat] = 0.65
        object_visuals[name] = {"rgb": rgb.tolist(), "finish": finish}

    pad_mat = model.material("pad_finish_mat").id
    pad_rgb = np.array([0.0, float(rng.uniform(0.7, 1.0)), float(rng.uniform(0.0, 0.25))])
    model.mat_rgba[pad_mat, :3] = pad_rgb
    model.mat_rgba[pad_mat, 3] = 0.45

    if mode == "lightness":
        if rng.random() < 0.5:
            model.vis.headlight.ambient[:] = rng.uniform(0.04, 0.12, size=3)
            model.vis.headlight.diffuse[:] = rng.uniform(0.20, 0.40, size=3)
            key_diff = rng.uniform(0.15, 0.35, size=3)
        else:
            model.vis.headlight.ambient[:] = rng.uniform(0.50, 0.70, size=3)
            model.vis.headlight.diffuse[:] = rng.uniform(0.85, 1.00, size=3)
            key_diff = rng.uniform(0.85, 1.00, size=3)
        model.vis.headlight.specular[:] = rng.uniform(0.0, 0.15, size=3)
    elif mode == "standard":
        model.vis.headlight.ambient[:] = np.array([0.30, 0.30, 0.30])
        model.vis.headlight.diffuse[:] = np.array([0.60, 0.60, 0.60])
        model.vis.headlight.specular[:] = np.zeros(3)
        key_diff = np.array([0.65, 0.65, 0.65])
    else:
        model.vis.headlight.ambient[:] = rng.uniform(0.15, 0.45, size=3)
        model.vis.headlight.diffuse[:] = rng.uniform(0.35, 0.85, size=3)
        model.vis.headlight.specular[:] = rng.uniform(0.0, 0.25, size=3)
        key_diff = rng.uniform(0.4, 0.9, size=3)
    light_id = model.light("task_key_light").id
    model.light_pos[light_id] = np.array(
        [rng.uniform(-0.2, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(1.2, 1.9)]
    )
    model.light_dir[light_id] = np.array([0.0, 0.0, -1.0])
    model.light_diffuse[light_id] = key_diff
    model.vis.rgba.haze[:3] = rng.uniform(0.05, 0.4, size=3)

    return {
        "table_texture": table_texture,
        "table_rgb": table_rgb.tolist(),
        "table_finish": table_finish,
        "object_visuals": object_visuals,
        "pad_rgb": pad_rgb.tolist(),
        "visual_mode": mode or "random",
    }


def apply_layout(model, data, layout, rng):
    apply_sizes(model, layout["stick_half"], layout["cube_half"])
    stick_z = TABLE_TOP + layout["stick_half"] + 0.001
    cube_z = TABLE_TOP + layout["cube_half"] + 0.001
    stick_quat = _yaw_quat(layout["stick_yaw"])
    if layout["stick_lying"]:
        stick_z = TABLE_TOP + STICK_RADIUS + 0.002
        stick_quat = _tip_quat(rng, layout["stick_yaw"])
    _set_free_pose(
        model,
        data,
        "tool_stick",
        [float(layout["stick_xy"][0]), float(layout["stick_xy"][1]), stick_z],
        stick_quat,
    )
    if layout["cube_present"]:
        _set_free_pose(
            model,
            data,
            "obj_cube",
            [float(layout["cube_xy"][0]), float(layout["cube_xy"][1]), cube_z],
            _yaw_quat(layout["cube_yaw"]),
        )
    else:
        _set_free_pose(model, data, "obj_cube", list(CUBE_PARK), _yaw_quat(0.0))
    _set_pad_xy(model, layout["pad_xy"])

    placed = []
    if layout.get("block_approach"):
        stick_xy = np.asarray(layout["stick_xy"], float)
        xy = np.array([stick_xy[0] - 0.07, stick_xy[1]])
        _set_free_pose(
            model,
            data,
            "junk_sphere",
            [xy[0], xy[1], TABLE_TOP + SPHERE_R + 0.001],
            _yaw_quat(0.0),
        )
        placed.append({"name": "junk_sphere", "xy": [float(xy[0]), float(xy[1])]})
    else:
        if rng.random() < 0.45:
            for _ in range(20):
                xy = np.array([rng.uniform(0.55, 0.78), rng.uniform(-0.20, 0.05)])
                if (
                    np.linalg.norm(xy - np.asarray(layout["stick_xy"])) > 0.10
                    and np.linalg.norm(xy - np.asarray(layout["cube_xy"])) > 0.10
                    and np.linalg.norm(xy - np.asarray(layout["pad_xy"])) > 0.12
                ):
                    _set_free_pose(
                        model,
                        data,
                        "junk_sphere",
                        [xy[0], xy[1], TABLE_TOP + SPHERE_R + 0.001],
                        _yaw_quat(0.0),
                    )
                    placed.append(
                        {"name": "junk_sphere", "xy": [float(xy[0]), float(xy[1])]}
                    )
                    break
            else:
                _set_free_pose(
                    model, data, "junk_sphere", list(SPHERE_PARK), _yaw_quat(0.0)
                )
        else:
            _set_free_pose(model, data, "junk_sphere", list(SPHERE_PARK), _yaw_quat(0.0))

    for i, name in enumerate(JUNK_NAMES):
        xy = layout["junk_xy"][i]
        if xy is None:
            park = JUNK_PARK[i]
            _set_free_pose(model, data, name, list(park), _yaw_quat(0.0))
        else:
            _set_free_pose(
                model,
                data,
                name,
                [float(xy[0]), float(xy[1]), TABLE_TOP + 0.03],
                _yaw_quat(0.0),
            )
            placed.append({"name": name, "xy": [float(xy[0]), float(xy[1])]})
    mujoco.mj_forward(model, data)
    return placed


def randomize_episode(model, data, kind, rng, layout_hint=None):
    layout = sample_layout(kind, rng, layout_hint=layout_hint)
    visuals = apply_visuals(model, rng, mode=layout.get("subtype") if kind == "success" else None)
    distractors = apply_layout(model, data, layout, rng)
    cube_xy = (
        [float(layout["cube_xy"][0]), float(layout["cube_xy"][1])]
        if layout["cube_present"]
        else [float(CUBE_PARK[0]), float(CUBE_PARK[1])]
    )
    return {
        "kind": kind,
        "layout_subtype": layout["subtype"],
        "required_type": "cube",
        "stick_xy": [float(layout["stick_xy"][0]), float(layout["stick_xy"][1])],
        "cube_xy": cube_xy,
        "pad_xy": [float(layout["pad_xy"][0]), float(layout["pad_xy"][1])],
        "stick_yaw": float(layout["stick_yaw"]),
        "cube_yaw": float(layout["cube_yaw"]),
        "stick_half": float(layout["stick_half"]),
        "cube_half": float(layout["cube_half"]),
        "stick_lying": bool(layout["stick_lying"]),
        "cube_present": bool(layout["cube_present"]),
        "distractors": distractors,
        "table_texture": visuals["table_texture"],
        "table_rgb": visuals["table_rgb"],
        "table_finish": visuals["table_finish"],
        "object_visuals": visuals["object_visuals"],
        "pad_rgb": visuals["pad_rgb"],
        "visual_mode": visuals["visual_mode"],
    }
