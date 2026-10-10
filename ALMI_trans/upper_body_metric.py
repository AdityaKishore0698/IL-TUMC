"""Objective upper-body measure for text-conditioned humanoid control: which hand waves, and how well.

ALMI judges upper-body success ("the robot waves the correct hand") by watching videos. This module
replaces that judgement with dynamic time warping (DTW) against the ALMI-X reference motions:

  * each arm's executed joint trajectory (4 joints, offsets from the default pose) is compared with the
    same arm in reference trajectories of every hand class (wave left / right / both);
  * DTW aligns the two motions in time (within a band) before measuring their difference, so a correct
    wave that starts late or runs slower is not penalised the way a frame-by-frame error would be;
  * the executed run is assigned the class whose k nearest references are closest (k-NN on DTW
    distance, summed over both arms); success = that class equals the commanded hand;
  * the mean DTW distance to the commanded class (rad per step) is the quality score;
  * a run whose nearest class is still farther than MAX_MATCH is "no clear wave": its arm motion is
    not like any reference wave (held-out ALMI-X waves score 0.29 median, 0.70 at the 90th percentile
    and 2.04 at most, at 400 steps), so it counts as a failure whatever the nearest class is.

Only the time before a fall is scored; a run that falls within MIN_STEPS is reported as unscorable.

Commands (run from ALMI_trans/):
  python upper_body_metric.py build --out output/arm_refs.npz          # on cir, from dataset/ALMI
  python upper_body_metric.py selftest --refs output/arm_refs.npz      # held-out accuracy on ALMI-X
  python upper_body_metric.py score --refs output/arm_refs.npz run1.npz [run2.npz ...]
The run files are written by play_trans.py / play_ol.py --log.
"""
import argparse
import os
import re
import sys

import numpy as np

HANDS = ("left", "right", "both")
LEFT_ARM, RIGHT_ARM = slice(13, 17), slice(17, 21)   # indices in the 21 joints
DOF_POS = slice(6, 27)                               # joint offsets inside the 71-dim observation
BAND = 50                                            # DTW band: 1 s at 50 Hz
MIN_STEPS = 50                                       # need at least 1 s upright to score a run
K = 5                                                # nearest references per class
MAX_MATCH = 2.0                                      # worst held-out ALMI-X wave (2.04): beyond this, no clear wave
REFS_PER_CLASS = 60                                  # references kept per class (distinct motions)


def hand_of(text):
    m = re.search(r"wave (left|right|both)", text)
    return m.group(1) if m else None


def dtw(x, y, band=BAND):
    """DTW distance between equal-length sequences x, y (T, d), Euclidean step cost, Sakoe-Chiba band,
    divided by T. Each row is solved in closed form: D[i, j] = S_j + min_{k<=j}(c_k - S_k), where c_k is
    the best predecessor from row i-1 and S the running sum of row i's costs."""
    n = len(x)
    assert len(y) == n
    prev = np.full(n + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(n, i + band)
        cost = np.linalg.norm(y[lo - 1:hi] - x[i - 1], axis=1)
        cand = np.minimum(prev[lo - 1:hi], prev[lo:hi + 1]) + cost
        s = np.cumsum(cost)
        row = np.full(n + 1, np.inf)
        row[lo:hi + 1] = s + np.minimum.accumulate(cand - s)
        prev = row
    return prev[n] / n


class References:
    def __init__(self, path, refs_per_class=REFS_PER_CLASS, seed=0, exclude_motions=()):
        z = np.load(path, allow_pickle=True)
        arms, lens, hands, motions = z["arms"], z["lens"], z["hands"], z["motions"]
        rng = np.random.default_rng(seed)
        self.items = {h: [] for h in HANDS}
        for h in HANDS:
            # one trajectory per distinct motion, so the references are not near-copies of each other
            idx = [i for i in rng.permutation(len(hands)) if hands[i] == h and motions[i] not in exclude_motions]
            seen, keep = set(), []
            for i in idx:
                if motions[i] not in seen:
                    seen.add(motions[i]); keep.append(i)
                if len(keep) == refs_per_class:
                    break
            self.items[h] = [arms[i, :lens[i]] for i in keep]

    def distances(self, arms):
        """arms: (T, 8) executed left+right arm offsets -> {hand: sorted per-reference distances}."""
        out = {}
        for h, refs in self.items.items():
            d = []
            for r in refs:
                n = min(len(arms), len(r))
                d.append(dtw(arms[:n, :4], r[:n, :4]) + dtw(arms[:n, 4:], r[:n, 4:]))
            out[h] = np.sort(d)
        return out

    def classify(self, arms, k=K):
        """-> (hand or "none", {hand: mean distance of its k nearest references})"""
        d = self.distances(arms)
        score = {h: float(np.mean(v[:k])) for h, v in d.items()}
        best = min(score, key=score.get)
        return (best if score[best] <= MAX_MATCH else "none"), score


def build(args):
    root = args.data
    names, texts = [], []
    for line in open(os.path.join(root, "train_ALMI.txt")):
        n = line.strip()
        try:
            cap = open(os.path.join(root, "texts", n + ".txt")).readline().split("#")[0]
        except OSError:
            continue
        if hand_of(cap):
            names.append(n); texts.append(cap)
    trajs = [np.load(os.path.join(root, "actions", n + ".npy"), allow_pickle=True).item()["obs"][:, DOF_POS]
             for n in names]
    T = max(len(t) for t in trajs)
    arms = np.zeros((len(trajs), T, 8), np.float32)
    for i, t in enumerate(trajs):
        arms[i, :len(t)] = np.concatenate([t[:, LEFT_ARM], t[:, RIGHT_ARM]], axis=1)
    np.savez_compressed(args.out, arms=arms, lens=np.array([len(t) for t in trajs]),
                        hands=np.array([hand_of(c) for c in texts]), texts=np.array(texts),
                        motions=np.array([n.split("-go_")[0].split("-keep")[0] for n in names]))
    print(f"wrote {args.out}: {len(names)} waving trajectories, " +
          ", ".join(f"{h} {sum(hand_of(c) == h for c in texts)}" for h in HANDS))


def selftest(args):
    """Classify held-out ALMI-X trajectories (motions never used as references)."""
    z = np.load(args.refs, allow_pickle=True)
    rng = np.random.default_rng(1)
    motions = np.unique(z["motions"])
    test_motions = set(rng.choice(motions, size=len(motions) // 5, replace=False))
    refs = References(args.refs, exclude_motions=test_motions)
    test = [i for i in rng.permutation(len(z["hands"])) if z["motions"][i] in test_motions]
    per = {h: [i for i in test if z["hands"][i] == h][:args.n] for h in HANDS}
    conf = {h: {g: 0 for g in HANDS} for h in HANDS}
    for h, idx in per.items():
        for i in idx:
            pred, _ = refs.classify(z["arms"][i, :min(z["lens"][i], args.steps)])
            conf[h][pred] += 1
    total = sum(len(v) for v in per.values())
    correct = sum(conf[h][h] for h in HANDS)
    print(f"held-out accuracy {correct}/{total} = {correct / total:.1%} "
          f"(first {args.steps} steps, {REFS_PER_CLASS} references per class, k={K})")
    print("true \\ predicted  " + "  ".join(f"{g:>5s}" for g in HANDS))
    for h in HANDS:
        print(f"{h:>16s}  " + "  ".join(f"{conf[h][g]:5d}" for g in HANDS))


def score(args):
    refs = References(args.refs)
    rows = []
    for path in args.runs:
        z = np.load(path, allow_pickle=True)
        text, q = str(z["text"]), z["qpos"]
        default = np.array(args.default, np.float32)
        off = q - default
        arms = np.concatenate([off[:, LEFT_ARM], off[:, RIGHT_ARM]], axis=1)
        fell = float(z["fell_at"])
        upright = int(round(fell / float(z["dt"]))) if fell >= 0 else len(arms)
        arms = arms[:upright]
        want = hand_of(text)
        survival = fell if fell >= 0 else len(z["qpos"]) * float(z["dt"])
        if len(arms) < MIN_STEPS:
            print(f"{os.path.basename(path)}: \"{text}\" -> unscorable (fell after {len(arms)} steps)")
            rows.append(dict(run=os.path.basename(path), text=text, asked=want, waves="unscorable", success=0,
                             quality=float("nan"), survival=survival))
            continue
        pred, sc = refs.classify(arms)
        print(f"{os.path.basename(path)}: \"{text}\" -> waves {pred} "
              f"({'correct' if pred == want else 'WRONG, asked ' + str(want)}); "
              f"DTW to left/right/both {sc['left']:.3f} / {sc['right']:.3f} / {sc['both']:.3f}; "
              f"scored {len(arms)} steps")
        rows.append(dict(run=os.path.basename(path), text=text, asked=want, waves=pred, success=int(pred == want),
                         quality=sc[want], survival=survival))
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    n = len(rows)
    print(f"SUMMARY {n} runs: upper-body success {sum(r['success'] for r in rows)}/{n}, "
          f"no clear wave {sum(r['waves'] == 'none' for r in rows)}, unscorable {sum(r['waves'] == 'unscorable' for r in rows)}, "
          f"mean DTW to the asked class {np.nanmean([r['quality'] for r in rows]):.3f}, "
          f"mean survival {np.mean([r['survival'] for r in rows]):.2f} s")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--data", default="dataset/ALMI"); b.add_argument("--out", required=True)
    s = sub.add_parser("selftest"); s.add_argument("--refs", required=True)
    s.add_argument("--n", type=int, default=100, help="held-out trajectories per class")
    s.add_argument("--steps", type=int, default=400)
    c = sub.add_parser("score"); c.add_argument("--refs", required=True); c.add_argument("runs", nargs="+")
    c.add_argument("--csv", default="", help="also write one row per run to this CSV")
    c.add_argument("--default", type=float, nargs=21, default=[0, -0.4, 0, 0.8, -0.4, 0, 0, -0.4, 0, 0.8, -0.4, 0, 0,
                                                                0.4, 0.2, 0, 0.3, 0.4, -0.2, 0, 0.3])
    a = p.parse_args()
    {"build": build, "selftest": selftest, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
