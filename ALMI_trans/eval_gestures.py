"""Does per-arm contrastive guidance generalise beyond waving? Protocol for salute, punch and shake hand
(run on cir, CPU, parallel, resumable). Companion of eval_protocol.py, which it reuses.

For every model x gesture caption x start: an 8 s MuJoCo run (play_trans.py --log), then
  * gesture success: exactly the arms the caption asks for perform the gesture (gesture_criteria.py; waving
    uses wave_criterion.py as in eval_protocol.py);
  * gesture success while upright (needs >= 2 s upright) -- separates "did not perform" from "fell";
  * walking direction (the paper's SR_low), 8 s survival, fall rate;
  * wall-clock time per run (subprocess), for the inference overhead of guidance.
Captions: every distinct ALMI-X caption of the gesture (41 locomotion commands x gesture phrases).

Models (policy G = text standardisation trained on wave, salute, punch and shake):
  G               : no guidance
  G+gesture       : per-arm contrastive guidance, s = 12, contrasts from gesture_guidance.py
  G+gesture+loco  : as G+gesture, arms without a contrast caption use the locomotion-only caption (exploratory)

Usage (from ALMI_trans/):
  python eval_gestures.py --gestures salute punch shake --models G G+gesture --seeds 0 1 2 --jobs 16 \
      --out output/gestures_<date>
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

import eval_protocol as ep

HERE = os.path.dirname(os.path.abspath(__file__))
G_POLICY = "output/export/policy_trans_cl20sl_tn_gest4_20261023_9fed470wip.pt"
GUIDE = ["--hand_guidance", "12", "--guidance_mode", "gesture"]
MODELS = {
    "G": ["--policy", G_POLICY],
    "G+gesture": ["--policy", G_POLICY, *GUIDE],
    "G+gesture+loco": ["--policy", G_POLICY, *GUIDE, "--contrast_fallback", "loco"],
    "A": ["--policy", ep.A_POLICY],
}


def commands(gesture):
    root = os.path.join(HERE, "dataset/ALMI")
    caps = set()
    for name in open(os.path.join(root, "train_ALMI.txt")).read().split():
        try:
            cap = open(os.path.join(root, "texts", name + ".txt")).readline().split("#")[0]
        except OSError:
            continue
        if gesture in cap:
            caps.add(cap)
    return sorted(caps)


def run(job):
    model, text, seed, path = job
    if os.path.exists(path):
        return path
    cmd = [sys.executable, "deploy/deploy_mujoco/play_trans.py", *MODELS[model], "--text", text, "--duration", "8",
           "--no_viewer", "--seed", str(seed), "--log", path]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED {model} | {text} | seed {seed}\n{r.stderr[-800:]}", flush=True)
    else:
        open(path + ".time", "w").write(f"{time.time() - t0:.2f}")
    return path


_kin = _cal = None


def score(job):
    global _kin, _cal
    import gesture_criteria as gc
    import gesture_features as gf
    import wave_criterion as wc
    if _kin is None:
        _kin = gf.ArmKinematics()
        _cal = json.load(open(gc.CALIBRATION))
        _cal["swing_min"] = json.load(open(wc.CALIBRATION))["swing_min"]
    model, text, seed, path = job
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    q, dt, fell = z["qpos"], float(z["dt"]), float(z["fell_at"])
    n_up = int(round(fell / dt)) if fell >= 0 else len(q)
    g = gc.gesture_of(text)
    expected = gc.expected_arms(text) if g != "wave" else (
        {"left", "right"} if "wave both" in text else {re.search(r"wave (left|right)", text).group(1)})

    def arms_performing(n_total):
        if g == "wave":
            feats = _kin.arm_features(q[:n_up])
            return {s for s, (h, ang) in feats.items() if wc.waving(h, ang, n_total, _cal["swing_min"])[0]}
        return gc.label(g, _kin.positions(q[:n_up]), n_total, _cal)[0]

    arms = arms_performing(len(q))
    arms_up = arms_performing(n_up) if n_up >= 100 else None
    vel = z["vel"][:n_up]
    v = vel.mean(0) if len(vel) else np.zeros(3)
    name = lambda s: "none" if not s else ("both" if len(s) == 2 else next(iter(s)))
    t = open(path + ".time").read() if os.path.exists(path + ".time") else "nan"
    return dict(model=model, gesture=g, text=text, seed=seed, expected=name(expected), performed=name(arms),
                success=int(arms == expected), performed_while_up=name(arms_up) if arms_up is not None else "fell",
                success_while_up=int(arms_up == expected) if arms_up is not None else 0,
                lower_ok=ep.direction_ok(text, v, n_up), full=int(fell < 0), fell=int(fell >= 0),
                survival=round(fell if fell >= 0 else len(q) * dt, 2), runtime_s=float(t))


def summarise(rows):
    lines = ["| Gesture | Model | Runs | Success (95 % CI) | By expected arm(s) | Success while upright | Not performed while upright | "
             "Fell | Walks as commanded | Upright 8 s | Run time (s) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in sorted({r["gesture"] for r in rows}):
        for m in sorted({r["model"] for r in rows}):
            r = [x for x in rows if x["model"] == m and x["gesture"] == g]
            if not r:
                continue
            n, k = len(r), sum(x["success"] for x in r)
            lo, hi = ep.wilson(k, n)
            per = ", ".join(f"{e} {sum(x['success'] for x in r if x['expected'] == e)}/{sum(x['expected'] == e for x in r)}"
                            for e in ("left", "right", "both") if any(x["expected"] == e for x in r))
            up = [x for x in r if x["performed_while_up"] != "fell"]
            notperf = sum(1 for x in up if not x["success_while_up"])
            lines.append(f"| {g} | {m} | {n} | {k}/{n} = {k / n:.0%} ({lo:.0%}–{hi:.0%}) | {per} | "
                         f"{sum(x['success_while_up'] for x in r)}/{n} | {notperf}/{n} | {sum(x['fell'] for x in r)}/{n} | "
                         f"{sum(x['lower_ok'] for x in r)}/{n} | {sum(x['full'] for x in r)}/{n} | "
                         f"{np.nanmedian([x['runtime_s'] for x in r]):.1f} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gestures", nargs="+", default=["salute", "punch", "shake"])
    p.add_argument("--models", nargs="+", default=["G", "G+gesture"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--max-commands", type=int, default=0, help="pilot: first N captions per gesture (0 = all)")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--out", required=True)
    p.add_argument("--score-only", action="store_true")
    a = p.parse_args()
    jobs = []
    for g in a.gestures:
        caps = commands(g)
        if a.max_commands:
            caps = caps[::max(1, len(caps) // a.max_commands)][:a.max_commands]
        for m in a.models:
            os.makedirs(os.path.join(HERE, a.out, m), exist_ok=True)
            for text in caps:
                for s in a.seeds:
                    jobs.append((m, text, s, os.path.join(HERE, a.out, m, f"{ep.slug(text)}_s{s}.npz")))
    print(f"{len(jobs)} runs", flush=True)
    with Pool(a.jobs) as pool:
        if not a.score_only:
            for i, _ in enumerate(pool.imap_unordered(run, jobs), 1):
                if i % 100 == 0:
                    print(f"  ran {i}/{len(jobs)}", flush=True)
        rows = [r for r in pool.map(score, jobs, chunksize=4) if r]
    out = os.path.join(HERE, a.out)
    with open(os.path.join(out, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    table = summarise(rows)
    open(os.path.join(out, "summary.md"), "w").write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
