"""Evaluation protocol for language-conditioned upper-body control (run on cir, CPU, many processes).

For every model x command x start: an 8 s MuJoCo run (play_trans.py / play_ol.py --log), then
  * upper body: does exactly the commanded arm wave (wave_criterion.py: hand at or above the shoulder,
    forearm swing >= half a typical ALMI-X wave, for more than half of the 8 s)?
  * motion quality: DTW distance of the arms to ALMI-X waves of the commanded hand (upper_body_metric.py)
  * lower body: does the movement direction match the command (the paper's SR_low)? Mean body-frame
    velocity while upright; every commanded axis must have the commanded sign and at least 0.05 m/s or
    rad/s; "keep standing" requires all three below 0.1
  * the same hand check over the upright time only (the paper judges the hand even when the robot later
    falls; needs >= 2 s upright, one 2 s window)
  * survival: time upright (pelvis above 0.5 m, torso tilt below 60 degrees), full 8 s or not

Commands: the 123 distinct waving captions of ALMI-X (41 locomotion commands x 3 hands), or --grid12.
Starts: --seeds (each perturbs the initial joint positions by up to +-0.05 rad; MuJoCo is deterministic).
Runs are skipped when their log exists, so an interrupted protocol resumes.

Usage (from ALMI_trans/):
  python eval_protocol.py --models CL-20sl CL-400sl OL A A+guidance --seeds 0 1 2 --jobs 70 --out output/protocol
  python eval_protocol.py ... --score-only          # rescore existing logs
"""
import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A_POLICY = "output/export/policy_trans_cl20sl_tn_20261001_f313925wip.pt"
GUIDE = ["--hand_guidance", "12", "--guidance_mode", "per_arm"]
H_POLICY = "output/export/policy_trans_cl20sl_tn_hand_20261014_629b6d1wip.pt"
MODELS = {
    "CL-20sl": dict(player="trans", args=["--policy", "output/export/policy_trans_cl20sl.pt"]),
    "CL-400sl": dict(player="trans", args=["--policy", "output/export/policy_trans_cl400sl.pt", "--seq_len", "400"]),
    "OL": dict(player="ol", args=["--trans", "output/ol_20260930_b128acc/almi_trans_ol_last.pth",
                                  "--vq", "output/vq_20260930_8a1wip/vq_net_last.pth"]),
    "A": dict(player="trans", args=["--policy", A_POLICY]),
    # final method: per-arm hand guidance on single-hand commands, none on "wave both"
    "A+guidance": dict(player="trans", args=["--policy", A_POLICY], single_hand=GUIDE),
    "C+guidance": dict(player="trans", args=["--policy", "output/export/policy_trans_cl20sl_tn_mir_20261010_e6ce4c9wip.pt",
                                             "--hand_guidance", "4"]),
    # H: explicit hand token, trained with hand-token and caption dropout
    "H": dict(player="trans", args=["--policy", H_POLICY]),
    "H+per-arm": dict(player="trans", args=["--policy", H_POLICY], single_hand=GUIDE),
    "H+cfg": dict(player="trans", args=["--policy", H_POLICY, "--hand_guidance", "2", "--guidance_mode", "cfg"]),
}
GRID12 = [f"Robot go {l} and wave {h}." for l in ("forward slowly", "backward moderately", "right fast", "left fast")
          for h in ("left", "right", "both")]


def spec_of(name):
    """MODELS entry, or a guidance sweep: 'H+cfg<s>' (classifier-free, arms) / 'H+per-arm<s>' (single-hand only)."""
    m = re.fullmatch(r"H\+cfg([\d.]+)", name)
    if m:
        return dict(player="trans", args=["--policy", H_POLICY, "--hand_guidance", m.group(1), "--guidance_mode", "cfg"])
    m = re.fullmatch(r"H\+per-arm([\d.]+)", name)
    if m:
        return dict(player="trans", args=["--policy", H_POLICY],
                    single_hand=["--hand_guidance", m.group(1), "--guidance_mode", "per_arm"])
    return MODELS[name]


def all_commands():
    z = np.load(os.path.join(HERE, "output/arm_refs.npz"), allow_pickle=True)
    return sorted(set(z["texts"].tolist()))


def slug(text):
    return re.sub(r"[^a-z]+", "_", text.lower().replace("robot ", "")).strip("_")


def run_one(job):
    model, text, seed, path = job
    if os.path.exists(path):
        return path
    spec = spec_of(model)
    script = "deploy/deploy_mujoco/play_trans.py" if spec["player"] == "trans" else "deploy/deploy_mujoco/play_ol.py"
    args = list(spec["args"])
    if "single_hand" in spec and "wave both" not in text:
        args += spec["single_hand"]
    cmd = [sys.executable, script, *args, "--text", text, "--duration", "8", "--no_viewer", "--seed", str(seed), "--log", path]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED {model} | {text} | seed {seed}\n{r.stderr[-800:]}", flush=True)
    return path


# ---------------------------------------------------------------------------------------------- scoring

def direction_ok(text, v, upright_steps):
    """The paper's lower-body success: the movement direction matches the command."""
    if upright_steps < 100:  # fell within 2 s: no meaningful direction
        return 0
    vx, vy, wz = v
    t = text.lower()
    if "keep standing" in t:
        return int(abs(vx) < 0.1 and abs(vy) < 0.1 and abs(wz) < 0.1)
    if "turn left" in t:
        return int(wz > 0.05)
    if "turn right" in t:
        return int(wz < -0.05)
    ok = True
    if "forward" in t:
        ok &= vx > 0.05
    if "backward" in t:
        ok &= vx < -0.05
    if re.search(r"\b(go|to the) left\b", t):
        ok &= vy > 0.05
    if re.search(r"\b(go|to the) right\b", t):
        ok &= vy < -0.05
    return int(ok)


_kin = _refs = _swing = None


def score_one(job):
    global _kin, _refs, _swing
    import wave_criterion as wc
    import upper_body_metric as ubm
    if _kin is None:
        _kin = wc.Kinematics()
        _refs = ubm.References(os.path.join(HERE, "output/arm_refs.npz"))
        _swing = json.load(open(wc.CALIBRATION))["swing_min"]
    model, text, seed, path = job
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    q, dt, fell = z["qpos"], float(z["dt"]), float(z["fell_at"])
    n_up = int(round(fell / dt)) if fell >= 0 else len(q)
    feats = _kin.arm_features(q[:n_up])
    wav = {s: wc.waving(h, a, len(q), _swing) for s, (h, a) in feats.items()}
    label = {(True, True): "both", (True, False): "left", (False, True): "right", (False, False): "none"}[
        (wav["left"][0], wav["right"][0])]
    asked = ubm.hand_of(text)
    # the paper's view: the right hand while the robot is still up (fraction of upright time, >= 2 s upright)
    wav_up = {s: wc.waving(h, a, n_up, _swing) for s, (h, a) in feats.items()} if n_up >= 100 else None
    label_up = ({(True, True): "both", (True, False): "left", (False, True): "right", (False, False): "none"}[
        (wav_up["left"][0], wav_up["right"][0])] if wav_up else "fell")
    vel = z["vel"][:n_up] if "vel" in z.files else np.zeros((0, 3))
    v = vel.mean(0) if len(vel) else np.zeros(3)
    off = q - wc.DEFAULT
    arms = np.concatenate([off[:n_up, 13:17], off[:n_up, 17:21]], axis=1)
    dtw = float("nan")
    if len(arms) >= ubm.MIN_STEPS:
        dtw = _refs.classify(arms)[1][asked]
    return dict(model=model, text=text, seed=seed, asked=asked, waves=label, upper_ok=int(label == asked),
                waves_while_up=label_up, upper_ok_while_up=int(label_up == asked),
                lower_ok=direction_ok(text, v, n_up), survival=round(fell if fell >= 0 else len(q) * dt, 2),
                full=int(fell < 0), vx=round(v[0], 3), vy=round(v[1], 3), wz=round(v[2], 3), dtw=round(dtw, 3),
                left_time=round(wav["left"][1], 2), right_time=round(wav["right"][1], 2))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarise(rows, models):
    lines = ["| Model | Runs | Upper body: commanded hand waves (95 % CI) | left / right / both | Lower body: direction | "
             "Hand correct while upright | Upright 8 s | Mean survival (s) | DTW to commanded wave (median) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        r = [x for x in rows if x["model"] == m]
        if not r:
            continue
        n, k = len(r), sum(x["upper_ok"] for x in r)
        lo, hi = wilson(k, n)
        per = " / ".join(f"{sum(x['upper_ok'] for x in r if x['asked'] == h) / max(1, sum(x['asked'] == h for x in r)):.0%}"
                         for h in ("left", "right", "both"))
        lines.append(f"| {m} | {n} | {k / n:.0%} ({lo:.0%}–{hi:.0%}) | {per} | "
                     f"{sum(x['lower_ok'] for x in r) / n:.0%} | {sum(x['upper_ok_while_up'] for x in r) / n:.0%} | "
                     f"{sum(x['full'] for x in r) / n:.0%} | "
                     f"{np.mean([x['survival'] for x in r]):.2f} | {np.nanmedian([x['dtw'] for x in r]):.2f} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["CL-20sl", "CL-400sl", "OL", "A", "A+guidance", "C+guidance"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--grid12", action="store_true", help="the 12-command grid instead of all 123 commands")
    p.add_argument("--jobs", type=int, default=64)
    p.add_argument("--out", default="output/protocol")
    p.add_argument("--score-only", action="store_true")
    a = p.parse_args()
    commands = GRID12 if a.grid12 else all_commands()
    jobs = []
    for m in a.models:
        os.makedirs(os.path.join(HERE, a.out, m), exist_ok=True)
        for text in commands:
            for s in a.seeds:
                jobs.append((m, text, s, os.path.join(HERE, a.out, m, f"{slug(text)}_s{s}.npz")))
    print(f"{len(jobs)} runs: {len(a.models)} models x {len(commands)} commands x {len(a.seeds)} starts", flush=True)
    with Pool(a.jobs) as pool:
        if not a.score_only:
            for i, _ in enumerate(pool.imap_unordered(run_one, jobs), 1):
                if i % 100 == 0:
                    print(f"  ran {i}/{len(jobs)}", flush=True)
        rows = [r for r in pool.map(score_one, jobs, chunksize=4) if r]
    out = os.path.join(HERE, a.out)
    with open(os.path.join(out, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    table = summarise(rows, a.models)
    open(os.path.join(out, "summary.md"), "w").write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
