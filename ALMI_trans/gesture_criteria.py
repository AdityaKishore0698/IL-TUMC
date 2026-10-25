"""Objective success criteria for the non-waving gestures of ALMI-X: salute, punch, shake hand.

All features are measured in the torso frame relative to the shoulder joint of each arm (MuJoCo forward
kinematics of the joint angles, gesture_features.ArmKinematics); hand = middle-finger knuckle.

Per arm, over sliding 2 s windows (0.25 s hop) of the full 8 s test (time after a fall counts as not performing):
  * salute     : the hand is within RADIUS of the reference salute hand position (forehead region) in more
                 than half of the windows (window median position);
  * shake hand : the hand is held out in front at waist-to-chest height (window median at least SHAKE_X
                 forward and between SHAKE_Z low and high relative to the shoulder) in more than half of the windows;
  * punch      : the hand extends forward at shoulder height at least once: its forward distance exceeds PUNCH_X
                 while its height is above PUNCH_Z (an extension ends when the hand retracts below PUNCH_X - 0.1 m).
All thresholds are set from the ALMI-X demonstrations only, before any model is evaluated (`calibrate`). A first
version (shake: within 0.15 m of the mean shake-hand position; punch: two extensions) passed only 32 % and 23 % of
the demonstrated active arms, because the shake-hand clips hold the hand at different places and most punch clips
contain one punch within a trajectory; it was replaced by the rules above before any evaluation.

A run succeeds when exactly the arms the caption asks for perform the gesture:
  salute -> right arm (every ALMI-X salute uses the right arm; the caption names no side)
  punch left / punch right -> that arm; punch alternately / keep punching forward -> both arms
  shake right hand -> right arm; shake both hands -> both arms

Usage (from ALMI_trans/, cir):
  python gesture_criteria.py calibrate        # writes gesture_criteria.json, prints reference pass rates
"""
import argparse
import json
import os
import re

import numpy as np

import gesture_features as gf
import wave_criterion as wc

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION = os.path.join(HERE, "gesture_criteria.json")
RADIUS = 0.15
PUNCH_Z = -0.30
SHAKE_X, SHAKE_Z = 0.30, (-0.65, -0.20)
WIN, HOP = gf.WIN, gf.HOP


def expected_arms(caption):
    c = caption.lower()
    if "salute" in c:
        return {"right"}
    if "alternately" in c or "keep punching" in c or "both" in c:
        return {"left", "right"}
    m = re.search(r"(punch|wave) (left|right)", c) or re.search(r"shake (left|right) hand", c)
    return {m.group(2) if m.lastindex == 2 else m.group(1)} if m else None


def gesture_of(caption):
    return gf.gesture_of(caption)


def mirror_y(p, side):
    """Express positions so that +y points to the body midline for either arm (left arm: midline is -y)."""
    p = np.array(p, float)
    if side == "left":
        p = p * np.array([1, -1, 1])
    return p


def window_medians(hand, n_total):
    out = []
    for s in range(0, max(1, n_total - WIN + 1), HOP):
        out.append(np.median(hand[s:s + WIN], 0) if s + WIN <= len(hand) else np.full(3, np.nan))
    return np.array(out)


def holds_near(hand, n_total, target, side):
    w = window_medians(hand, n_total)
    d = np.linalg.norm(mirror_y(w, side) - np.array(target), axis=1)
    frac = float(np.mean(np.nan_to_num(d, nan=np.inf) <= RADIUS))
    return frac > 0.5, frac


def punch_count(hand, punch_x):
    out, n = True, 0
    for x, z in zip(hand[:, 0], hand[:, 2]):
        if out and x >= punch_x and z >= PUNCH_Z:
            n, out = n + 1, False
        elif not out and x < punch_x - 0.1:
            out = True
    return n


def arm_performs(gesture, hand, n_total, side, cal):
    """-> (performs, score) for one arm."""
    if gesture == "salute":
        return holds_near(hand, n_total, cal["salute_target"], side)
    if gesture == "shake":
        w = mirror_y(window_medians(hand, n_total), side)
        inside = (w[:, 0] >= SHAKE_X) & (w[:, 2] >= SHAKE_Z[0]) & (w[:, 2] <= SHAKE_Z[1])
        frac = float(np.mean(inside))  # NaN windows (after a fall) compare False
        return frac > 0.5, frac
    if gesture == "punch":
        k = punch_count(hand, cal["punch_x"])
        return k >= 1, float(k)
    raise ValueError(gesture)


def label(gesture, positions, n_total, cal):
    res = {s: arm_performs(gesture, positions[s]["hand"], n_total, s, cal) for s in ("left", "right")}
    arms = {s for s, (ok, _) in res.items() if ok}
    return arms, {s: sc for s, (_, sc) in res.items()}


def calibrate(args):
    root = os.path.join(HERE, "dataset/ALMI")
    kin = gf.ArmKinematics()
    data = {"salute": [], "shake": [], "punch": []}
    rng = np.random.default_rng(0)
    names = open(os.path.join(root, "train_ALMI.txt")).read().split()
    for name in rng.permutation(names):
        try:
            cap = open(os.path.join(root, "texts", name + ".txt")).readline().split("#")[0]
        except OSError:
            continue
        g = gesture_of(cap)
        if g not in data or len(data[g]) >= args.n:
            continue
        q = np.load(os.path.join(root, "actions", name + ".npy"), allow_pickle=True).item()["obs"][:, 6:27] + wc.DEFAULT
        data[g].append((cap, kin.positions(q), len(q)))
        if all(len(v) >= args.n for v in data.values()):
            break
    cal = {}
    # reference hand positions: median over the active arm (midline-mirrored) of single-arm demonstrations
    for g, phrase in (("salute", None), ("shake", "shake right hand")):
        pts = []
        for cap, pos, n in data[g]:
            if phrase and phrase not in cap:
                continue
            for s in expected_arms(cap):
                pts.append(np.median(mirror_y(pos[s]["hand"], s), 0))
        cal[f"{g}_target"] = [float(v) for v in np.median(pts, 0)]
    # punch: forward reach (at shoulder height) that best separates punching from resting arms
    act, rest = [], []
    for cap, pos, n in data["punch"]:
        if not re.search(r"punch (left|right)", cap):
            continue
        for s in ("left", "right"):
            h = pos[s]["hand"]
            reach = float(h[h[:, 2] >= PUNCH_Z, 0].max()) if np.any(h[:, 2] >= PUNCH_Z) else -1.0
            (act if s in expected_arms(cap) else rest).append(reach)
    act, rest = np.array(act), np.array(rest)
    cands = np.linspace(0.3, 0.7, 401)
    score = [((act >= t).mean() + (rest < t).mean()) / 2 for t in cands]
    cal["punch_x"] = float(cands[int(np.argmax(score))])
    cal["radius"], cal["punch_z"], cal["n_per_gesture"] = RADIUS, PUNCH_Z, args.n
    cal["shake_x"], cal["shake_z"] = SHAKE_X, list(SHAKE_Z)
    # reference pass rates under the final criteria
    report = {}
    for g, items in data.items():
        tp = fp = n_act = n_rest = 0
        for cap, pos, n in items:
            arms, _ = label(g, pos, n, cal)
            exp = expected_arms(cap)
            for s in ("left", "right"):
                if s in exp:
                    n_act += 1
                    tp += s in arms
                else:
                    n_rest += 1
                    fp += s in arms
        exact = np.mean([label(g, pos, n, cal)[0] == expected_arms(cap) for cap, pos, n in items])
        report[g] = dict(active_arms_pass=tp / max(1, n_act), resting_arms_pass=fp / max(1, n_rest), exact=float(exact),
                         trajectories=len(items))
    cal["reference_check"] = report
    json.dump(cal, open(CALIBRATION, "w"), indent=1)
    print(json.dumps(cal, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--n", type=int, default=300, help="demonstrations per gesture")
    a = p.parse_args()
    calibrate(a)
