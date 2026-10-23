"""Compare the ALMI-X demonstrations of "wave left", "wave right" and "wave both".

Question: is the right-hand problem in the training data? For every waving trajectory of the training set
(run on cir, reads dataset/ALMI), per arm and per trajectory:
  * source motion clip (KIT capture) and trajectory length,
  * hand height above the shoulder (median; share of time at or above the shoulder) and shoulder pitch,
  * forearm swing amplitude (as in wave_criterion.py) and swing frequency (dominant frequency of the swing),
  * whether the arm meets the wave criterion (wave_criterion.py),
and per trajectory the robot's stability while it was recorded:
  * torso tilt (norm of the horizontal projected gravity), base angular speed, leg joint speed.
Waving arm of "wave left" is compared with the waving arm of "wave right" (and the resting arms likewise).

Usage (from ALMI_trans/, cir): python analyse_wave_data.py --out ../results/<date>/wave_data
Writes <out>.csv (one row per trajectory) and <out>.md (summary table).
"""
import argparse
import csv
import json
import os
import re

import numpy as np

import wave_criterion as wc

DOF = slice(6, 27)


def swing_frequency(angle, dt=0.02):
    x = angle - np.convolve(np.pad(angle, (25, 24), mode="edge"), np.ones(50) / 50, "valid")
    if len(x) < 100:
        return float("nan")
    f = np.fft.rfftfreq(len(x), dt)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    band = (f >= 0.3) & (f <= 5.0)
    return float(f[band][np.argmax(p[band])])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset/ALMI")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    swing_min = json.load(open(wc.CALIBRATION))["swing_min"]
    kin = wc.Kinematics()
    rows = []
    for name in open(os.path.join(a.data, "train_ALMI.txt")).read().split():
        try:
            cap = open(os.path.join(a.data, "texts", name + ".txt")).readline().split("#")[0]
        except OSError:
            continue
        m = re.search(r"wave (left|right|both)", cap)
        if not m:
            continue
        hand = m.group(1)
        obs = np.load(os.path.join(a.data, "actions", name + ".npy"), allow_pickle=True).item()["obs"]
        q = obs[:, DOF] + wc.DEFAULT
        feats = kin.arm_features(q)
        clip = re.search(r"(KIT_\d+_[A-Za-z0-9_]+?_poses|KIT_\d+_Trial_\d+)", name)
        row = dict(name=name, caption=cap, hand=hand, clip=clip.group(1) if clip else name.split("-go_")[0],
                   steps=len(q),
                   tilt=float(np.median(np.linalg.norm(obs[:, 3:5], axis=1))),
                   ang_speed=float(np.median(np.linalg.norm(obs[:, 0:3], axis=1))),
                   leg_speed=float(np.median(np.abs(obs[:, 27:39]).mean(1))))
        for side, (h, ang) in feats.items():
            waving, frac, swing, height = wc.waving(h, ang, len(q), swing_min)
            sp = q[:, 13] if side == "left" else q[:, 17]  # shoulder pitch (absolute)
            row.update({f"{side}_role": "wave" if hand in (side, "both") else "rest",
                        f"{side}_height": float(np.median(h)), f"{side}_above": float(np.mean(h >= 0)),
                        f"{side}_swing": float(swing), f"{side}_freq": swing_frequency(ang),
                        f"{side}_shoulder_pitch": float(np.median(sp)), f"{side}_meets_criterion": int(waving),
                        f"{side}_time": float(frac)})
        rows.append(row)
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def stat(vals):
        v = np.array([x for x in vals if np.isfinite(x)])
        return f"{np.median(v):.3f} [{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}]"

    lines = ["| Measure (median [25 %, 75 %]) | wave left: left arm | wave right: right arm | wave both: left / right arm |",
             "|---|---|---|---|"]
    L = [r for r in rows if r["hand"] == "left"]
    R = [r for r in rows if r["hand"] == "right"]
    B = [r for r in rows if r["hand"] == "both"]
    lines.append(f"| Trajectories / source clips | {len(L)} / {len(set(r['clip'] for r in L))} | "
                 f"{len(R)} / {len(set(r['clip'] for r in R))} | {len(B)} / {len(set(r['clip'] for r in B))} |")
    for key, label in (("height", "Hand height above shoulder (m)"), ("above", "Share of time hand at/above shoulder"),
                       ("shoulder_pitch", "Shoulder pitch (rad)"), ("swing", "Forearm swing (rad)"),
                       ("freq", "Swing frequency (Hz)"), ("time", "Share of windows meeting the wave criterion")):
        lines.append(f"| {label} | {stat([r['left_' + key] for r in L])} | {stat([r['right_' + key] for r in R])} | "
                     f"{stat([r['left_' + key] for r in B])} / {stat([r['right_' + key] for r in B])} |")
    lines.append(f"| Waving arm meets the wave criterion | {np.mean([r['left_meets_criterion'] for r in L]):.0%} | "
                 f"{np.mean([r['right_meets_criterion'] for r in R]):.0%} | "
                 f"{np.mean([r['left_meets_criterion'] for r in B]):.0%} / {np.mean([r['right_meets_criterion'] for r in B]):.0%} |")
    lines.append(f"| Resting arm meets the wave criterion | {np.mean([r['right_meets_criterion'] for r in L]):.0%} | "
                 f"{np.mean([r['left_meets_criterion'] for r in R]):.0%} | – |")
    for key, label in (("tilt", "Torso tilt (projected gravity xy)"), ("ang_speed", "Base angular speed (scaled)"),
                       ("leg_speed", "Leg joint speed (scaled)")):
        lines.append(f"| {label} | {stat([r[key] for r in L])} | {stat([r[key] for r in R])} | {stat([r[key] for r in B])} |")
    table = "\n".join(lines)
    open(a.out + ".md", "w").write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
