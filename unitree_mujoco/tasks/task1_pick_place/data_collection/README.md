# Task 1 VLA data collection

Headless collector for pick-and-place demonstrations and flagged boundary failures.

```bash
/Users/huyuxuan/g1_env/bin/python collect_task1.py --out datasets/task1/smoke --n 10 --seed 0
```

## Episode kinds

| Kind | Policy | `is_success` | `failure_reason` |
|------|--------|--------------|------------------|
| `success` | `Task1HandController` | true if cube on pad | null |
| `space_constraint` | approach then retract | false | `space_constraint` |
| `kinematics_limit` | hold current pose (action = 0) | false | `kinematics_limit` |

Do **not** mix `is_success==false` retract/hold trajectories into behavior cloning without the flags. Use `reward` / `phase` on all kinds for a process value or event head.

## Files

Each `ep_XXXXXX/` contains:

- `episode.json` — instruction, `is_success`, `failure_reason`, spawn, `return`, `phase_returns`
- `episode.hdf5` — 10 Hz rows: images, qpos/qvel, ee, `action` (joint delta), `phase`, `reward`, `reward_terms`, `events`
