"""Kinematic features of arm gestures (torso frame, MuJoCo forward kinematics), shared by the gesture
criteria and the reference-data analysis.

Per arm and control step: hand (middle-finger knuckle) position relative to the shoulder joint, elbow position,
and the elbow joint angle. From these, per 2 s window (0.25 s hop): median hand position, hand oscillation
(RMS after removing a 1 s moving average) along each axis, and median elbow angle.

Usage for the reference analysis (from ALMI_trans/, cir):
  python gesture_features.py --out ../results/<date>/gesture_data
"""
import argparse
import csv
import os
import re

import mujoco
import numpy as np

import wave_criterion as wc

WIN, HOP, DETREND = 100, 12, 50
ELBOW = {"left": 16, "right": 20}


class ArmKinematics(wc.Kinematics):
    def __init__(self):
        super().__init__()
        body = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.bodies = {s: dict(shoulder=body(f"{s}_shoulder_pitch_link"), elbow=body(f"{s}_elbow_pitch_link"),
                               hand=body(("L" if s == "left" else "R") + "_middle_proximal")) for s in ("left", "right")}

    def positions(self, joints):
        """joints: (T, 21) absolute angles -> {side: dict(hand=(T,3), elbow=(T,3)) relative to the shoulder}."""
        out = {s: dict(hand=[], elbow=[]) for s in self.bodies}
        self.d.qpos[:] = 0
        self.d.qpos[3] = 1.0
        for q in joints:
            self.d.qpos[7:] = q
            mujoco.mj_kinematics(self.m, self.d)
            R = self.d.xmat[self.torso].reshape(3, 3)
            for s, b in self.bodies.items():
                sh = self.d.xpos[b["shoulder"]]
                out[s]["hand"].append((self.d.xpos[b["hand"]] - sh) @ R)
                out[s]["elbow"].append((self.d.xpos[b["elbow"]] - sh) @ R)
        return {s: {k: np.array(v) for k, v in d.items()} for s, d in out.items()}


def window_features(pos, elbow_angle, n_total):
    """Per 2 s window: median hand xyz (3), hand oscillation RMS xyz (3), median elbow angle (1).
    Windows past the data (after a fall) are NaN."""
    k = np.ones(DETREND) / DETREND
    osc = np.stack([pos[:, i] - np.convolve(np.pad(pos[:, i], (DETREND // 2, DETREND - 1 - DETREND // 2), mode="edge"), k, "valid")
                    for i in range(3)], 1) if len(pos) else np.zeros((0, 3))
    rows = []
    for s in range(0, max(1, n_total - WIN + 1), HOP):
        if s + WIN <= len(pos):
            rows.append(np.concatenate([np.median(pos[s:s + WIN], 0), np.sqrt((osc[s:s + WIN] ** 2).mean(0)),
                                        [np.median(elbow_angle[s:s + WIN])]]))
        else:
            rows.append(np.full(7, np.nan))
    return np.array(rows)


def gesture_of(caption):
    for g in ("salute", "punch", "shake", "wave"):
        if g in caption:
            return g
    return None


def active_arms(caption):
    """Arms the caption asks to move (None if the caption names no side)."""
    c = caption.lower()
    if "alternately" in c or "both" in c:
        return {"left", "right"}
    m = re.search(r"(punch|shake|wave) (left|right)", c)
    return {m.group(2)} if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset/ALMI")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    kin = ArmKinematics()
    rows = []
    for name in open(os.path.join(a.data, "train_ALMI.txt")).read().split():
        try:
            cap = open(os.path.join(a.data, "texts", name + ".txt")).readline().split("#")[0]
        except OSError:
            continue
        g = gesture_of(cap)
        if g not in ("salute", "punch", "shake"):
            continue
        q = np.load(os.path.join(a.data, "actions", name + ".npy"), allow_pickle=True).item()["obs"][:, 6:27] + wc.DEFAULT
        pos = kin.positions(q)
        clip = re.search(r"(KIT_\d+_[A-Za-z0-9_]+?_poses|KIT_\d+_Trial_\d+)", name)
        row = dict(name=name, caption=cap, gesture=g, clip=clip.group(1) if clip else name.split("-go_")[0], steps=len(q))
        for s in ("left", "right"):
            w = window_features(pos[s]["hand"], q[:, ELBOW[s]], len(q))
            hand = pos[s]["hand"]
            row.update({f"{s}_x_med": float(np.median(hand[:, 0])), f"{s}_y_med": float(np.median(hand[:, 1])),
                        f"{s}_z_med": float(np.median(hand[:, 2])), f"{s}_x_max": float(hand[:, 0].max()),
                        f"{s}_z_max": float(hand[:, 2].max()), f"{s}_reach_max": float(np.linalg.norm(hand, axis=1).max()),
                        f"{s}_osc_x": float(np.nanmedian(w[:, 3])), f"{s}_osc_y": float(np.nanmedian(w[:, 4])),
                        f"{s}_osc_z": float(np.nanmedian(w[:, 5])), f"{s}_elbow_med": float(np.median(q[:, ELBOW[s]])),
                        f"{s}_elbow_min": float(q[:, ELBOW[s]].min())})
        rows.append(row)
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} trajectories written to {a.out}.csv")


if __name__ == "__main__":
    main()
