"""Is the MuJoCo H1-2 model used for evaluation left-right symmetric?

1. Static check: every left body/joint/actuator/collision geom is compared with its right counterpart
   (masses, inertias, centres of mass and positions mirrored in y, joint ranges mirrored for roll/yaw joints,
   damping, armature, friction, actuator limits, collision geometry).
2. Dynamic check, no policy involved: the robot stands with every joint PD-controlled to the default pose
   (deployment gains), and one arm replays the arm motion of an ALMI-X wave. Each wave is run twice: as recorded
   on its own arm, and mirrored onto the other arm. In a symmetric model the two responses are mirror images:
   base roll and lateral position change sign, everything else is equal. Also reports how far the robot tilts
   and whether it falls, for left-arm and right-arm waves.

Usage (from ALMI_trans/, cir): python symmetry_test.py --out ../results/<date>/symmetry [--waves 40]
"""
import argparse
import json
import os
import re

import mujoco
import numpy as np
import yaml

from utils.mirror import JOINT_PERM, JOINT_SIGN

CFG = "deploy/deploy_mujoco/configs/h1_2_21dof_trans.yaml"


def counterpart(name):
    for a, b in (("left_", "right_"), ("right_", "left_"), ("L_", "R_"), ("R_", "L_")):
        if name.startswith(a):
            return b + name[len(a):]
    return None


def static_check(m):
    issues, rows = [], []
    flip = np.array([1, -1, 1])
    for b in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        c = counterpart(n or "")
        if not n or not (n.startswith("left_") or n.startswith("L_")):
            continue
        cb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c)
        if cb < 0:
            issues.append(f"body {n}: no counterpart {c}")
            continue
        d = dict(body=n,
                 mass=abs(m.body_mass[b] - m.body_mass[cb]),
                 inertia=float(np.abs(np.sort(m.body_inertia[b]) - np.sort(m.body_inertia[cb])).max()),
                 com=float(np.abs(m.body_ipos[b] * flip - m.body_ipos[cb]).max()),
                 pos=float(np.abs(m.body_pos[b] * flip - m.body_pos[cb]).max()),
                 ngeom=int(m.body_geomnum[b]) - int(m.body_geomnum[cb]),
                 ncollide=int(sum((m.geom_contype[g] | m.geom_conaffinity[g]) > 0 for g in range(m.ngeom) if m.geom_bodyid[g] == b))
                 - int(sum((m.geom_contype[g] | m.geom_conaffinity[g]) > 0 for g in range(m.ngeom) if m.geom_bodyid[g] == cb)))
        rows.append(d)
    jrows = []
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not n or not (n.startswith("left_") or n.startswith("L_")):
            continue
        cj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, counterpart(n))
        axis = m.jnt_axis[j]
        mirrored = bool(abs(axis[0]) > 0.5 or abs(axis[2]) > 0.5)  # roll (x) and yaw (z) joints flip sign
        lo, hi = m.jnt_range[j]
        rlo, rhi = m.jnt_range[cj]
        exp = (-hi, -lo) if mirrored else (lo, hi)
        dof, cdof = m.jnt_dofadr[j], m.jnt_dofadr[cj]
        jrows.append(dict(joint=n, range=float(max(abs(exp[0] - rlo), abs(exp[1] - rhi))),
                          damping=abs(m.dof_damping[dof] - m.dof_damping[cdof]),
                          armature=abs(m.dof_armature[dof] - m.dof_armature[cdof]),
                          friction=abs(m.dof_frictionloss[dof] - m.dof_frictionloss[cdof])))
    arows = []
    for a in range(m.nu):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or mujoco.mj_id2name(
            m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0])
        if not n or not n.startswith("left_"):
            continue
        ca = next((k for k in range(m.nu) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, k) or
                   mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[k, 0])) == counterpart(n)), None)
        if ca is None:
            continue
        arows.append(dict(actuator=n, ctrlrange=float(np.abs(np.abs(m.actuator_ctrlrange[a]) - np.abs(m.actuator_ctrlrange[ca])).max()),
                          forcerange=float(np.abs(np.abs(m.actuator_forcerange[a]) - np.abs(m.actuator_forcerange[ca])).max()),
                          gear=float(abs(m.actuator_gear[a, 0] - m.actuator_gear[ca, 0]))))
    return rows, jrows, arows, issues


def wave_response(m, cfg, arm_traj, side, seconds=6.0):
    """Stand on PD at the default pose; `side` arm follows arm_traj (T, 4 offsets). Returns roll, lateral pos, tilt."""
    d = mujoco.MjData(m)
    m.opt.timestep = cfg["simulation_dt"]
    kps, kds = np.array(cfg["kps"]), np.array(cfg["kds"])
    default = np.array(cfg["default_angles"])
    d.qpos[7:] = default
    d.qpos[2] = 1.03  # start standing (pelvis height of the default pose), then settle
    dec = cfg["control_decimation"]
    arm = slice(13, 17) if side == "left" else slice(17, 21)
    roll, lat, tilt = [], [], []
    settle = 100  # 2 s of standing before the arm moves
    for k in range(settle + int(seconds / (dec * cfg["simulation_dt"]))):
        target = default.copy()
        i = k - settle
        if i >= 0:
            target[arm] = default[arm] + arm_traj[min(i, len(arm_traj) - 1)]
        for _ in range(dec):
            d.ctrl[:] = (target - d.qpos[7:]) * kps - d.qvel[6:] * kds
            mujoco.mj_step(m, d)
        if i >= 0:
            w, x, y, z = d.qpos[3:7]
            roll.append(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
            lat.append(d.qpos[1])
            gx, gy = 2 * (-z * x + w * y), -2 * (z * y + w * x)
            tilt.append(np.hypot(gx, gy))
    return np.array(roll), np.array(lat), np.array(tilt), bool(d.qpos[2] < 0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--waves", type=int, default=40)
    a = p.parse_args()
    cfg = yaml.safe_load(open(CFG))
    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])

    bodies, joints, acts, issues = static_check(m)
    lines = ["# Symmetry test of the MuJoCo H1-2 model", "", "## 1. Static check (left minus mirrored right)", ""]
    lines += [f"* Bodies compared: {len(bodies)}; joints: {len(joints)}; actuators: {len(acts)}; missing counterparts: {len(issues)}"]
    for key, unit in (("mass", "kg"), ("inertia", "kg m²"), ("com", "m"), ("pos", "m")):
        worst = max(bodies, key=lambda r: r[key])
        lines.append(f"* Largest body {key} difference: {worst[key]:.3g} {unit} ({worst['body']})")
    lines.append(f"* Bodies whose geom count differs: {[r['body'] for r in bodies if r['ngeom']] or 'none'}; "
                 f"whose collision-geom count differs: {[(r['body'], r['ncollide']) for r in bodies if r['ncollide']] or 'none'}")
    for key in ("range", "damping", "armature", "friction"):
        worst = max(joints, key=lambda r: r[key])
        lines.append(f"* Largest joint {key} difference: {worst[key]:.3g} ({worst['joint']})")
    if acts:
        for key in ("ctrlrange", "forcerange", "gear"):
            worst = max(acts, key=lambda r: r[key])
            lines.append(f"* Largest actuator {key} difference: {worst[key]:.3g} ({worst['actuator']})")
    lines += [f"* {x}" for x in issues]
    kp, kd = np.array(cfg["kps"]), np.array(cfg["kds"])
    lines.append(f"* Deployment PD gains symmetric: kp {np.allclose(kp[JOINT_PERM], kp)}, kd {np.allclose(kd[JOINT_PERM], kd)}")

    # 2. dynamic check with recorded single-hand waves
    z = np.load("output/arm_refs.npz", allow_pickle=True)
    A, L, H = z["arms"], z["lens"], z["hands"]
    rng = np.random.default_rng(0)
    res = []
    for hand in ("left", "right"):
        idx = [i for i in rng.permutation(len(H)) if H[i] == hand][:a.waves // 2]
        for i in idx:
            own = A[i, :L[i], :4] if hand == "left" else A[i, :L[i], 4:]
            other = "right" if hand == "left" else "left"
            # mirror the 4 arm joints: pitch and elbow keep sign, roll and yaw flip
            mir = own * np.array([1, -1, -1, 1])
            r1, y1, t1, f1 = wave_response(m, cfg, own, hand)
            r2, y2, t2, f2 = wave_response(m, cfg, mir, other)
            n = min(len(r1), len(r2))
            res.append(dict(hand=hand, roll_asym=float(np.abs(r1[:n] + r2[:n]).max()), roll_mag=float(np.abs(r1[:n]).max()),
                            lat_asym=float(np.abs((y1[:n] - y1[0]) + (y2[:n] - y2[0])).max()),
                            lat_mag=float(np.abs(y1[:n] - y1[0]).max()),
                            tilt_own=float(t1.max()), tilt_mirrored=float(t2.max()), fell_own=f1, fell_mirrored=f2))
    lines += ["", "## 2. Dynamic check: a recorded wave on its own arm vs mirrored onto the other arm", "",
              f"Robot standing on PD at the default pose; {len(res)} ALMI-X single-hand waves, 6 s each.", "",
              "| | Recorded wave on its own arm | Same wave mirrored onto the other arm |", "|---|---|---|"]
    for hand in ("left", "right"):
        r = [x for x in res if x["hand"] == hand]
        other = "right" if hand == "left" else "left"
        lines.append(f"| {hand} waves (n={len(r)}): max torso tilt, median | {hand} arm: {np.median([x['tilt_own'] for x in r]):.3f} | "
                     f"{other} arm: {np.median([x['tilt_mirrored'] for x in r]):.3f} |")
        lines.append(f"| {hand} waves: falls | {sum(x['fell_own'] for x in r)} | {sum(x['fell_mirrored'] for x in r)} |")
    lines += ["", f"* Base roll, |own + mirrored| (0 if symmetric): median {np.median([x['roll_asym'] for x in res]):.4f} rad, "
              f"max {np.max([x['roll_asym'] for x in res]):.4f} rad; roll magnitude median {np.median([x['roll_mag'] for x in res]):.4f} rad",
              f"* Lateral drift, |own + mirrored|: median {np.median([x['lat_asym'] for x in res]):.4f} m, max "
              f"{np.max([x['lat_asym'] for x in res]):.4f} m; drift magnitude median {np.median([x['lat_mag'] for x in res]):.4f} m"]
    lw = [x["tilt_own"] for x in res if x["hand"] == "left"] + [x["tilt_mirrored"] for x in res if x["hand"] == "right"]
    rw = [x["tilt_own"] for x in res if x["hand"] == "right"] + [x["tilt_mirrored"] for x in res if x["hand"] == "left"]
    lines.append(f"* All runs with the LEFT arm waving: max tilt median {np.median(lw):.3f}; with the RIGHT arm waving: {np.median(rw):.3f}")
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out + ".md", "w").write(text + "\n")
    json.dump(dict(bodies=bodies, joints=joints, actuators=acts, issues=issues, dynamic=res), open(a.out + ".json", "w"), indent=1)
    print(text)


if __name__ == "__main__":
    main()
