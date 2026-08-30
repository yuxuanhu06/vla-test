"""Layout, visual, pad, and distractor randomization for Task 2 collection."""

import numpy as np
import mujoco

from .schema_task2 import OBJECT_TYPES


BODY_OF = {
    "tri_prism": "obj_tri_prism",
    "cylinder": "obj_cylinder",
    "cube": "obj_cube",
    "rect_prism": "obj_rect_prism",
}
MAT_OF = {
    "tri_prism": "obj_tri_prism_mat",
    "cylinder": "obj_cylinder_mat",
    "cube": "obj_cube_mat",
    "rect_prism": "obj_rect_prism_mat",
}

TABLE_TOP = 0.85
OBJECT_HALF_Z = 0.055
PAD_Z = 0.855
START_CELL = np.array([0.34, 0.26])
SPHERE_PARK = (1.70, 0.86, 0.04)
JUNK_PARK = ((1.70, 0.70, 0.03), (1.78, 0.70, 0.03))
JUNK_NAMES = ("junk_0", "junk_1")
SPHERE_R = 0.028


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


def _shuffle_types(rng):
    types = list(OBJECT_TYPES)
    rng.shuffle(types)
    return types


def _avoid_start_cell(xy):
    if np.linalg.norm(xy - START_CELL) < 0.05:
        xy = xy + np.array([0.06, -0.06])
    return xy


def _slots_to_objects(types, xys, reachable, standing, yaws):
    objects = {}
    for typ, xy, reach, stand, yaw in zip(types, xys, reachable, standing, yaws):
        objects[typ] = {
            "xy": [float(xy[0]), float(xy[1])],
            "yaw": float(yaw),
            "reachable": bool(reach),
            "standing": bool(stand),
            "present": True,
            "slot": "near" if reach else "far",
        }
    return objects


def _swap_into_slot(objects, type_hint, pred):
    """Move type_hint onto a pose that already satisfies pred."""
    if type_hint not in objects or pred(objects[type_hint]):
        return
    for other, rec in objects.items():
        if other == type_hint or not pred(rec):
            continue
        for key in ("xy", "yaw", "reachable", "standing", "slot"):
            objects[type_hint][key], objects[other][key] = (
                objects[other][key],
                objects[type_hint][key],
            )
        return


def _pick_required(kind, objects, rng, type_hint=None):
    standing = [t for t, o in objects.items() if o["standing"] and o["reachable"]]
    fallen = [t for t, o in objects.items() if not o["standing"]]
    far = [t for t, o in objects.items() if not o["reachable"]]
    near = [t for t, o in objects.items() if o["reachable"] and o["standing"]]

    if kind == "tipped":
        if type_hint:
            _swap_into_slot(objects, type_hint, lambda o: not o["standing"])
        pool = [t for t, o in objects.items() if not o["standing"]] or list(objects)
    elif kind == "kinematics_limit":
        if type_hint:
            _swap_into_slot(objects, type_hint, lambda o: not o["reachable"])
        pool = [t for t, o in objects.items() if not o["reachable"]] or list(objects)
    else:
        if type_hint:
            _swap_into_slot(
                objects,
                type_hint,
                lambda o: o["reachable"] and o["standing"],
            )
        pool = [t for t, o in objects.items() if o["reachable"] and o["standing"]] or standing or list(objects)

    if type_hint in pool:
        return type_hint
    return str(rng.choice(np.array(pool)))


def _working_row(rng):
    types = _shuffle_types(rng)
    ys = np.array([0.09, 0.17, 0.25, 0.33]) + rng.uniform(-0.015, 0.015, size=4)
    xs = 0.40 + rng.uniform(-0.02, 0.02, size=4)
    xys = [_avoid_start_cell(np.array([xs[i], ys[i]])) for i in range(4)]
    yaws = rng.uniform(-0.2, 0.2, size=4)
    return _slots_to_objects(types, xys, [True] * 4, [True] * 4, yaws), "working_row"


def _grid_2x2(rng):
    types = _shuffle_types(rng)
    slots = [
        (0.32, 0.12, True),
        (0.32, 0.28, True),
        (0.48, 0.12, False),
        (0.48, 0.28, False),
    ]
    xys = []
    reach = []
    for x, y, front in slots:
        jitter = rng.uniform(-0.015, 0.015, size=2)
        xy = _avoid_start_cell(np.array([x, y]) + jitter)
        xys.append(xy)
        reach.append(front)
    yaws = rng.uniform(-0.15, 0.15, size=4)
    objs = _slots_to_objects(types, xys, reach, [True] * 4, yaws)
    for typ, front in zip(types, reach):
        objs[typ]["slot"] = "front" if front else "back"
    return objs, "grid_2x2"


def _staggered(rng):
    types = _shuffle_types(rng)
    slots = [
        (0.32, 0.12, True),
        (0.46, 0.20, False),
        (0.32, 0.30, True),
        (0.46, 0.36, False),
    ]
    xys, reach = [], []
    for x, y, clear in slots:
        xy = _avoid_start_cell(np.array([x, y]) + rng.uniform(-0.012, 0.012, size=2))
        xys.append(xy)
        reach.append(clear)
    yaws = rng.uniform(-0.15, 0.15, size=4)
    objs = _slots_to_objects(types, xys, reach, [True] * 4, yaws)
    return objs, "staggered_two_row"


def _too_close(rng):
    types = _shuffle_types(rng)
    ys = np.array([0.14, 0.185, 0.23, 0.275]) + rng.uniform(-0.005, 0.005, size=4)
    xys = [np.array([0.40, y]) for y in ys]
    yaws = rng.uniform(-0.1, 0.1, size=4)
    return _slots_to_objects(types, xys, [True] * 4, [True] * 4, yaws), "too_close"


def _mixed_far(rng):
    types = _shuffle_types(rng)
    close = [
        _avoid_start_cell(np.array([0.38, 0.10]) + rng.uniform(-0.015, 0.015, size=2)),
        _avoid_start_cell(np.array([0.36, 0.22]) + rng.uniform(-0.015, 0.015, size=2)),
    ]
    far = [
        np.array([0.55, 0.37]) + rng.uniform(-0.02, 0.02, size=2),
        np.array([0.38, -0.34]) + rng.uniform(-0.02, 0.02, size=2),
    ]
    xys = close + far
    reach = [True, True, False, False]
    yaws = rng.uniform(-0.15, 0.15, size=4)
    objs = _slots_to_objects(types, xys, reach, [True] * 4, yaws)
    for i, typ in enumerate(types):
        objs[typ]["slot"] = "close" if reach[i] else "far"
    return objs, "mixed_far"


def _arc_small(rng):
    types = _shuffle_types(rng)
    # Slight +X bow so the -X lane and start cell (0.34, 0.26) stay empty.
    base = [
        (0.40, 0.09),
        (0.42, 0.17),
        (0.42, 0.25),
        (0.40, 0.33),
    ]
    xys = [_avoid_start_cell(np.array(p) + rng.uniform(-0.008, 0.008, size=2)) for p in base]
    yaws = rng.uniform(-0.12, 0.12, size=4)
    return _slots_to_objects(types, xys, [True] * 4, [True] * 4, yaws), "arc_small"


def _arc_large(rng):
    types = _shuffle_types(rng)
    # ends unreachable, middle two reachable
    base = [
        (0.38, -0.32),
        (0.36, 0.14),
        (0.36, 0.24),
        (0.42, 0.37),
    ]
    xys = [np.array(p) + rng.uniform(-0.012, 0.012, size=2) for p in base]
    xys[1] = _avoid_start_cell(xys[1])
    xys[2] = _avoid_start_cell(xys[2])
    reach = [False, True, True, False]
    yaws = rng.uniform(-0.15, 0.15, size=4)
    objs = _slots_to_objects(types, xys, reach, [True] * 4, yaws)
    for i, typ in enumerate(types):
        objs[typ]["slot"] = "end" if not reach[i] else "middle"
    return objs, "arc_large"


def _too_far_one(rng):
    types = _shuffle_types(rng)
    xys = [
        _avoid_start_cell(np.array([0.40, 0.10])),
        _avoid_start_cell(np.array([0.40, 0.20])),
        _avoid_start_cell(np.array([0.40, 0.30])),
        np.array([0.42, 0.37]),
    ]
    reach = [True, True, True, False]
    yaws = rng.uniform(-0.15, 0.15, size=4)
    return _slots_to_objects(types, xys, reach, [True] * 4, yaws), "too_far"


def _mixed_tipped(rng):
    types = _shuffle_types(rng)
    # Standing on the working row with a wide gap; fallen parked off the -X lanes.
    stand_xy = [
        _avoid_start_cell(np.array([0.40, 0.12]) + rng.uniform(-0.01, 0.01, size=2)),
        _avoid_start_cell(np.array([0.40, 0.30]) + rng.uniform(-0.01, 0.01, size=2)),
    ]
    fall_xy = [
        np.array([0.58, 0.10]) + rng.uniform(-0.01, 0.01, size=2),
        np.array([0.58, 0.32]) + rng.uniform(-0.01, 0.01, size=2),
    ]
    xys = stand_xy + fall_xy
    standing = [True, True, False, False]
    yaws = rng.uniform(-0.12, 0.12, size=4)
    objs = _slots_to_objects(types, xys, [True] * 4, standing, yaws)
    return objs, "mixed_tipped"


def sample_layout(kind, rng, layout_hint=None, type_hint=None):
    builders = {
        "working_row": _working_row,
        "grid_2x2": _grid_2x2,
        "staggered_two_row": _staggered,
        "too_close": _too_close,
        "mixed_far": _mixed_far,
        "arc_small": _arc_small,
        "arc_large": _arc_large,
        "too_far": _too_far_one,
        "mixed_tipped": _mixed_tipped,
        "blocked_approach": _working_row,
    }
    if layout_hint in builders:
        name = layout_hint
    elif kind == "tipped":
        name = "mixed_tipped"
    elif kind == "neighbor_collision":
        name = "too_close"
    elif kind == "kinematics_limit":
        name = str(rng.choice(["mixed_far", "arc_large", "too_far"]))
    elif kind == "space_constraint":
        name = str(rng.choice(["too_close", "blocked_approach"]))
    elif kind == "wrong_object":
        name = "working_row"
    else:
        name = str(
            rng.choice(
                [
                    "working_row",
                    "grid_2x2",
                    "staggered_two_row",
                    "arc_small",
                    "mixed_far",
                    "mixed_tipped",
                    "arc_large",
                ]
            )
        )

    objects, subtype = builders[name](rng)
    if name == "blocked_approach":
        subtype = "blocked_approach"
    if kind == "wrong_object":
        required = type_hint if type_hint in objects else str(rng.choice(np.array(OBJECT_TYPES)))
        objects[required]["present"] = False
        objects[required]["reachable"] = False
        objects[required]["xy"] = [1.70, 0.62]
    else:
        required = _pick_required(kind, objects, rng, type_hint=type_hint)
    return {
        "objects": objects,
        "layout_subtype": subtype,
        "required_type": required,
        "reachable_types": [t for t, o in objects.items() if o["reachable"]],
        "unreachable_types": [t for t, o in objects.items() if not o["reachable"]],
        "standing_types": [t for t, o in objects.items() if o["standing"]],
        "fallen_types": [t for t, o in objects.items() if not o["standing"]],
    }


def sample_pad(kind, objects, required_type, rng, layout_subtype):
    if kind == "space_constraint" and layout_subtype in ("too_close",) and rng.random() < 0.35:
        subtype = "jammed"
    elif kind == "space_constraint" and layout_subtype == "blocked_approach":
        subtype = "working"
    elif kind == "kinematics_limit" and rng.random() < 0.25:
        subtype = "far"
    elif kind == "space_constraint" and rng.random() < 0.4:
        subtype = "jammed"
    else:
        subtype = "working"

    if objects.get(required_type, {}).get("present", True):
        req_xy = np.array(objects[required_type]["xy"])
    else:
        present = [np.array(o["xy"]) for o in objects.values() if o.get("present", True)]
        req_xy = present[0] if present else np.array([0.40, 0.20])
    if subtype == "jammed":
        pad = req_xy + np.array([0.02, -0.02])
        return pad, "jammed", False
    if subtype == "far":
        pad = np.array([0.72, 0.34])
        return pad, "far", False

    for _ in range(40):
        pad = np.array(
            [rng.uniform(0.40, 0.50), rng.uniform(-0.18, -0.08)]
        )
        if np.linalg.norm(pad - req_xy) < 0.12:
            continue
        blocked = False
        for rec in objects.values():
            oxy = np.array(rec["xy"])
            if np.linalg.norm(pad - oxy) < 0.12:
                blocked = True
                break
            if abs(pad[1] - oxy[1]) < 0.06 and oxy[0] - 0.14 < pad[0] < oxy[0]:
                blocked = True
                break
        if not blocked:
            return pad, "working", True
    return np.array([0.44, -0.12]), "working", True


def apply_layout(model, data, objects, rng):
    for typ, rec in objects.items():
        xy = rec["xy"]
        if not rec.get("present", True):
            _set_free_pose(
                model,
                data,
                BODY_OF[typ],
                [1.70, 0.62, 0.03],
                _yaw_quat(0.0),
            )
            continue
        if rec["standing"]:
            z = TABLE_TOP + OBJECT_HALF_Z + 0.001
            quat = _yaw_quat(rec["yaw"])
        else:
            z = TABLE_TOP + 0.011 + 0.002
            quat = _tip_quat(rng, rec["yaw"])
        _set_free_pose(
            model,
            data,
            BODY_OF[typ],
            [xy[0], xy[1], z],
            quat,
        )


def apply_sphere(model, data, kind, objects, required_type, layout_subtype, rng):
    placed = []
    use_blocker = kind == "space_constraint" and layout_subtype == "blocked_approach"
    use_decor = (not use_blocker) and kind in ("success", "wrong_object") and rng.random() < 0.55
    if use_blocker:
        req = np.array(objects[required_type]["xy"])
        xy = np.array([req[0] - 0.07, req[1]])
        _set_free_pose(
            model,
            data,
            "junk_sphere",
            [xy[0], xy[1], TABLE_TOP + SPHERE_R + 0.001],
            _yaw_quat(0.0),
        )
        placed.append({"name": "junk_sphere", "xy": [float(xy[0]), float(xy[1])]})
    elif use_decor:
        for _ in range(20):
            xy = np.array([rng.uniform(0.55, 0.78), rng.uniform(-0.20, 0.05)])
            ok = True
            for rec in objects.values():
                if np.linalg.norm(xy - np.array(rec["xy"])) < 0.10:
                    ok = False
                    break
            if ok:
                _set_free_pose(
                    model,
                    data,
                    "junk_sphere",
                    [xy[0], xy[1], TABLE_TOP + SPHERE_R + 0.001],
                    _yaw_quat(0.0),
                )
                placed.append({"name": "junk_sphere", "xy": [float(xy[0]), float(xy[1])]})
                break
        else:
            _set_free_pose(model, data, "junk_sphere", list(SPHERE_PARK), _yaw_quat(0.0))
    else:
        _set_free_pose(model, data, "junk_sphere", list(SPHERE_PARK), _yaw_quat(0.0))

    for i, name in enumerate(JUNK_NAMES):
        park = JUNK_PARK[i]
        _set_free_pose(model, data, name, list(park), _yaw_quat(0.0))
    return placed


def _distinct_rgbs(rng, n=4, min_dist=0.35):
    colors = []
    for _ in range(80):
        if len(colors) == n:
            break
        rgb = rng.uniform(0.15, 0.95, size=3)
        rgb[int(rng.integers(0, 3))] = float(rng.uniform(0.7, 1.0))
        if all(float(np.linalg.norm(rgb - c)) >= min_dist for c in colors):
            colors.append(rgb)
    while len(colors) < n:
        colors.append(rng.uniform(0.2, 0.9, size=3))
    return colors


def apply_visuals(model, rng):
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

    colors = _distinct_rgbs(rng)
    object_visuals = {}
    for typ, rgb in zip(OBJECT_TYPES, colors):
        mat = model.material(MAT_OF[typ]).id
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
        object_visuals[typ] = {"rgb": rgb.tolist(), "finish": finish}

    model.vis.headlight.ambient[:] = rng.uniform(0.15, 0.45, size=3)
    model.vis.headlight.diffuse[:] = rng.uniform(0.35, 0.85, size=3)
    model.vis.headlight.specular[:] = rng.uniform(0.0, 0.25, size=3)
    light_id = model.light("task_key_light").id
    model.light_pos[light_id] = np.array(
        [rng.uniform(-0.2, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(1.2, 1.9)]
    )
    model.light_dir[light_id] = np.array([0.0, 0.0, -1.0])
    model.light_diffuse[light_id] = rng.uniform(0.4, 0.9, size=3)
    model.vis.rgba.haze[:3] = rng.uniform(0.05, 0.4, size=3)

    return {
        "table_texture": table_texture,
        "table_rgb": table_rgb.tolist(),
        "table_finish": table_finish,
        "object_visuals": object_visuals,
    }


def randomize_episode(model, data, kind, rng, layout_hint=None, type_hint=None):
    visuals = apply_visuals(model, rng)
    layout = sample_layout(kind, rng, layout_hint=layout_hint, type_hint=type_hint)
    apply_layout(model, data, layout["objects"], rng)
    pad_xy, pad_subtype, pad_reachable = sample_pad(
        kind, layout["objects"], layout["required_type"], rng, layout["layout_subtype"]
    )
    _set_pad_xy(model, pad_xy)
    distractors = apply_sphere(
        model,
        data,
        kind,
        layout["objects"],
        layout["required_type"],
        layout["layout_subtype"],
        rng,
    )
    mujoco.mj_forward(model, data)
    objects_out = {}
    for typ, rec in layout["objects"].items():
        vis = visuals["object_visuals"][typ]
        objects_out[typ] = {
            **rec,
            "rgb": vis["rgb"],
            "finish": vis["finish"],
        }
    return {
        "kind": kind,
        "layout_subtype": layout["layout_subtype"],
        "required_type": layout["required_type"],
        "reachable_types": layout["reachable_types"],
        "unreachable_types": layout["unreachable_types"],
        "standing_types": layout["standing_types"],
        "fallen_types": layout["fallen_types"],
        "missing_types": [
            t for t, o in objects_out.items() if not o.get("present", True)
        ],
        "objects": objects_out,
        "pad_xy": [float(pad_xy[0]), float(pad_xy[1])],
        "pad_subtype": pad_subtype,
        "pad_reachable": bool(pad_reachable),
        "distractors": distractors,
        "table_texture": visuals["table_texture"],
        "table_rgb": visuals["table_rgb"],
        "table_finish": visuals["table_finish"],
    }
