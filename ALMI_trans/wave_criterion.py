"""Is the robot waving the commanded hand? An explicit criterion, checked with forward kinematics.

Definition (thesis evaluation criterion): an arm waves correctly when
  1. its hand is raised to at least shoulder height,
  2. its forearm swings side to side, with a swing large enough to count as a wave, and
  3. both hold for most of the 8 s test (more than half of the time).
A run succeeds when exactly the commanded arm(s) wave: "wave left" = left arm only, "wave right" = right arm
only, "wave both" = both arms.

Measured per control step in the torso frame (MuJoCo forward kinematics of the logged joint angles):
  * height  = hand (middle-finger knuckle) height minus shoulder height;
  * swing   = side-to-side angle of the forearm (elbow -> hand) in the torso's frontal plane, after removing
              its 1 s moving average; the swing amplitude is the RMS of that angle over a 2 s window.
Sliding 2 s windows (0.25 s hop) over the full test: a window counts when the median height is >= 0 and the
swing amplitude is >= SWING_MIN. The arm waves if more than half of the windows count. Time after a fall
counts as not waving.

SWING_MIN is set on ALMI-X alone (no human labels): half the median swing of the waving arm in the single-hand
reference waves, i.e. "at least half as much swing as a typical demonstrated wave". (The value that merely
separates waving from resting arms, about 0.07 rad, also passes small arm jitter that a viewer does not call a
wave.) Human labels are only used to check the result.

Usage (from ALMI_trans/):
  python wave_criterion.py calibrate --refs output/arm_refs.npz
  python wave_criterion.py score run1.npz [run2.npz ...] [--csv out.csv]
"""
import argparse
import csv
import json
import os
import re

import mujoco
import numpy as np

XML = "deploy/robots/h1_2/scene_21dof.xml"
DEFAULT = np.array([0, -0.4, 0, 0.8, -0.4, 0, 0, -0.4, 0, 0.8, -0.4, 0, 0,
                    0.4, 0.2, 0, 0.3, 0.4, -0.2, 0, 0.3], np.float32)
ARMS = {"left": slice(13, 17), "right": slice(17, 21)}
CALIBRATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wave_criterion.json")
WIN, HOP, DETREND = 100, 12, 50  # 2 s windows, 0.25 s hop, 1 s moving average (50 Hz)


class Kinematics:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        body = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
        # hand point: the middle-finger knuckle (the wrist alone sits about 10 cm below the hand)
        self.ids = {s: (body(f"{s}_shoulder_pitch_link"), body(f"{s}_elbow_pitch_link"),
                        body(("L" if s == "left" else "R") + "_middle_proximal")) for s in ARMS}
        self.torso = body("torso_link")

    def arm_features(self, joints):
        """joints: (T, 21) absolute joint angles -> {side: (height (T,), forearm side angle (T,))}."""
        out = {s: ([], []) for s in ARMS}
        self.d.qpos[:] = 0
        self.d.qpos[3] = 1.0  # fixed, upright base: features are in the torso frame anyway
        for q in joints:
            self.d.qpos[7:] = q
            mujoco.mj_kinematics(self.m, self.d)
            R = self.d.xmat[self.torso].reshape(3, 3)
            o = self.d.xpos[self.torso]
            for s, (sh, el, wr) in self.ids.items():
                p_sh, p_el, p_wr = ((self.d.xpos[b] - o) @ R for b in (sh, el, wr))
                fore = p_wr - p_el
                out[s][0].append(p_wr[2] - p_sh[2])
                out[s][1].append(np.arctan2(fore[1], fore[2]))  # lateral tilt of the forearm, frontal plane
        return {s: (np.array(h), np.unwrap(np.array(a))) for s, (h, a) in out.items()}


def windows(height, angle, n_total):
    """Per 2 s window: (median height, swing amplitude); windows past the data count as not waving."""
    k = np.ones(DETREND) / DETREND
    pad = np.pad(angle, (DETREND // 2, DETREND - 1 - DETREND // 2), mode="edge")
    swing = angle - np.convolve(pad, k, "valid")
    res = []
    for s in range(0, max(1, n_total - WIN + 1), HOP):
        if s + WIN <= len(angle):
            res.append((np.median(height[s:s + WIN]), np.sqrt(np.mean(swing[s:s + WIN] ** 2))))
        else:
            res.append((-np.inf, 0.0))
    return np.array(res)


def waving(height, angle, n_total, swing_min):
    w = windows(height, angle, n_total)
    good = (w[:, 0] >= 0) & (w[:, 1] >= swing_min)
    return good.mean() > 0.5, good.mean(), np.median(w[:, 1]), np.median(w[:, 0])


def calibrate(args):
    z = np.load(args.refs, allow_pickle=True)
    arms, lens, hands = z["arms"], z["lens"], z["hands"]
    kin = Kinematics()
    rng = np.random.default_rng(0)
    idx = [i for i in rng.permutation(len(hands)) if hands[i] in ("left", "right")][:args.n]
    wave_amp, rest_amp, wave_up, rest_up = [], [], [], []
    for i in idx:
        q = np.tile(DEFAULT, (min(lens[i], 400), 1))
        q[:, 13:17] += arms[i, :len(q), :4]
        q[:, 17:21] += arms[i, :len(q), 4:]
        f = kin.arm_features(q)
        for s, (h, a) in f.items():
            w = windows(h, a, len(q))
            amp, up = np.median(w[:, 1]), np.mean(w[:, 0] >= 0)
            (wave_amp if s == hands[i] else rest_amp).append(amp)
            (wave_up if s == hands[i] else rest_up).append(up)
    wave_amp, rest_amp = np.array(wave_amp), np.array(rest_amp)
    t = float(0.5 * np.median(wave_amp))
    score = [((wave_amp >= t).mean() + (rest_amp < t).mean()) / 2]
    print(f"{len(idx)} single-hand reference waves")
    print(f"  waving arm: swing median {np.median(wave_amp):.3f} rad, hand above shoulder in {np.mean(wave_up):.0%} of windows")
    print(f"  resting arm: swing median {np.median(rest_amp):.3f} rad, hand above shoulder in {np.mean(rest_up):.0%} of windows")
    print(f"  SWING_MIN = {t:.3f} rad (half the median waving swing); waving arms above it "
          f"{(wave_amp >= t).mean():.0%}, resting arms below it {(rest_amp < t).mean():.0%}")
    json.dump({"swing_min": t, "balanced_accuracy": max(score), "n": len(idx)}, open(CALIBRATION, "w"), indent=1)


def score(args):
    t = json.load(open(CALIBRATION))["swing_min"]
    kin = Kinematics()
    rows = []
    for path in args.runs:
        z = np.load(path, allow_pickle=True)
        text, q = str(z["text"]), z["qpos"]
        fell = float(z["fell_at"])
        upright = int(round(fell / float(z["dt"]))) if fell >= 0 else len(q)
        f = kin.arm_features(q[:upright])
        res = {s: waving(h, a, len(q), t) for s, (h, a) in f.items()}
        label = {(True, True): "both", (True, False): "left", (False, True): "right", (False, False): "none"}[
            (res["left"][0], res["right"][0])]
        want = re.search(r"wave (left|right|both)", text).group(1)
        rows.append(dict(run=os.path.basename(path), text=text, asked=want, waves=label, success=int(label == want),
                         left_time=round(res["left"][1], 2), right_time=round(res["right"][1], 2),
                         left_swing=round(res["left"][2], 3), right_swing=round(res["right"][2], 3),
                         left_height=round(res["left"][3], 3), right_height=round(res["right"][3], 3),
                         survival=fell if fell >= 0 else len(q) * float(z["dt"])))
        if args.verbose:
            r = rows[-1]
            print(f"{r['run']}: asked {want}, waves {label} | left {r['left_time']:.0%} of time, swing "
                  f"{r['left_swing']}, height {r['left_height']} | right {r['right_time']:.0%}, swing "
                  f"{r['right_swing']}, height {r['right_height']}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = len(rows)
    print(f"SUMMARY {n} runs: correct wave {sum(r['success'] for r in rows)}/{n}; " + ", ".join(
        f"{h} {sum(r['success'] for r in rows if r['asked'] == h)}/{sum(r['asked'] == h for r in rows)}"
        for h in ("left", "right", "both")) + f"; upright 8 s {sum(r['survival'] >= 7.99 for r in rows)}/{n}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate"); c.add_argument("--refs", required=True); c.add_argument("--n", type=int, default=600)
    s = sub.add_parser("score"); s.add_argument("runs", nargs="+"); s.add_argument("--csv", default="")
    s.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    {"calibrate": calibrate, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
