"""Diagnostic for E19 (no training): how much left/right information survives each text-standardisation scheme?

For every distinct training caption of the four gestures, the CLIP embedding is computed exactly as in training
(CLIP ViT-B/32 on the GPU, fp16 weights) and standardised with
  * raw     : no standardisation (the paper),
  * A       : the statistics stored in model A's checkpoint (123 waving captions),
  * G       : the statistics stored in model G's checkpoint (410 captions of four gestures),
  * family  : statistics computed separately per gesture family (wave, punch, shake, salute),
  * hier    : family statistics for the within-family part plus G's statistics for the family mean:
              z = (e - mu_f) / sigma_f + (mu_f - mu) / sigma, so that families stay apart.
For each scheme it reports, on standardised embeddings:
  * hand separation: distance between captions that differ only in the hand phrase (same locomotion command),
    divided by the median distance between captions that differ only in the locomotion command (same hand phrase);
    larger = left/right more visible relative to the rest of the command;
  * gesture separation: nearest-family-centroid accuracy (leave-one-out) and distance between families relative
    to the within-family spread; the scheme must keep the gestures apart.
This is a property of the model inputs, not of the trained models; it shows whether a scheme can preserve the
hand signal before any training.

Usage (from ALMI_trans/, cir): python text_norm_diagnostic.py --out ../results/<date>/text_norm_diagnostic
"""
import argparse
import itertools
import json
import os
import re

import clip
import numpy as np
import torch

FAMILIES = ("wave", "punch", "shake", "salute")
HAND_PAIRS = {  # captions that differ only in the hand phrase
    "wave": [("wave left", "wave right"), ("wave left", "wave both"), ("wave right", "wave both")],
    "punch": [("punch left", "punch right"),
              ("punch left", "punch alternately with right and left hand"),
              ("punch right", "punch alternately with right and left hand")],
    "shake": [("shake right hand", "shake both hands")],
}


def family(caption):
    for f in FAMILIES:
        if f in caption:
            return f
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = "dataset/ALMI"
    caps = set()
    for name in open(os.path.join(root, "train_ALMI.txt")).read().split():
        try:
            c = open(os.path.join(root, "texts", name + ".txt")).readline().split("#")[0]
        except OSError:
            continue
        if family(c):
            caps.add(c)
    caps = sorted(caps)
    model, _ = clip.load("pretrained/ViT-B-32.pt", device=dev, jit=False)
    clip.model.convert_weights(model)  # as in train_almi_cl_20sl.py
    with torch.no_grad():
        E = torch.cat([model.encode_text(clip.tokenize(caps[i:i + 256], truncate=True).to(dev)).float()
                       for i in range(0, len(caps), 256)]).cpu().numpy()
    fam = np.array([family(c) for c in caps])
    split = [c.rstrip(".").partition(" and ") for c in caps]
    loco = np.array([s[0] for s in split])
    phrase = np.array([s[2] for s in split])

    def stats(X):
        return X.mean(0), np.clip(X.std(0, ddof=1), 1e-3, None)

    st = lambda path: torch.load(path, map_location="cpu")["trans"]
    sa = st("output/cl20sl_tn_20261001_f313925wip/almi_trans_cl_20sl_last.pth")
    sg = st("output/cl20sl_tn_gest4_20261023_9fed470wip/almi_trans_cl_20sl_last.pth")
    muA, sdA = sa["trans_base.text_mean"].numpy(), sa["trans_base.text_std"].numpy()
    muG, sdG = sg["trans_base.text_mean"].numpy(), sg["trans_base.text_std"].numpy()
    # check: G's stored statistics equal the statistics recomputed here (same captions, same CLIP)
    mu_all, sd_all = stats(E)
    check = dict(G_mean_max_abs_diff=float(np.abs(mu_all - muG).max()), G_std_max_rel_diff=float(np.abs(sd_all / sdG - 1).max()),
                 n_captions=len(caps), per_family={f: int((fam == f).sum()) for f in FAMILIES})
    fstats = {f: stats(E[fam == f]) for f in FAMILIES}

    schemes = {"raw": E, "A": (E - muA) / sdA, "G": (E - muG) / sdG}
    Zf = np.zeros_like(E)
    Zh = np.zeros_like(E)
    for f in FAMILIES:
        m = fam == f
        mu_f, sd_f = fstats[f]
        Zf[m] = (E[m] - mu_f) / sd_f
        Zh[m] = (E[m] - mu_f) / sd_f + (mu_f - muG) / sdG
    schemes["family"], schemes["hier"] = Zf, Zh

    idx = {(l, ph): i for i, (l, ph) in enumerate(zip(loco, phrase))}
    report = {"check": check}
    for name, Z in schemes.items():
        r = {}
        for f, pairs in HAND_PAIRS.items():
            hand_d = [np.linalg.norm(Z[idx[(l, x)]] - Z[idx[(l, y)]]) for x, y in pairs for l in set(loco)
                      if (l, x) in idx and (l, y) in idx]
            loco_d = []
            for ph in {x for pr in pairs for x in pr}:
                rows = [i for i in range(len(caps)) if phrase[i] == ph]
                loco_d += [np.linalg.norm(Z[i] - Z[j]) for i, j in itertools.combinations(rows, 2)]
            cos = [float(Z[idx[(l, x)]] @ Z[idx[(l, y)]] / (np.linalg.norm(Z[idx[(l, x)]]) * np.linalg.norm(Z[idx[(l, y)]])))
                   for x, y in pairs[:1] for l in set(loco) if (l, x) in idx and (l, y) in idx]
            r[f] = dict(hand_vs_loco_distance=float(np.median(hand_d) / np.median(loco_d)), first_pair_cosine=float(np.median(cos)))
        # gesture separation: leave-one-out nearest family centroid
        correct = 0
        for i in range(len(caps)):
            cents = {f: Z[(fam == f) & (np.arange(len(caps)) != i)].mean(0) for f in FAMILIES}
            correct += min(cents, key=lambda f: np.linalg.norm(Z[i] - cents[f])) == fam[i]
        between = np.median([np.linalg.norm(Z[fam == f].mean(0) - Z[fam == g].mean(0)) for f, g in itertools.combinations(FAMILIES, 2)])
        within = np.median([np.linalg.norm(Z[i] - Z[fam == fam[i]].mean(0)) for i in range(len(caps))])
        r["gesture_nearest_centroid_accuracy"] = correct / len(caps)
        r["gesture_between_over_within"] = float(between / within)
        report[name] = r
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out + ".json", "w"), indent=1)
    lines = ["| Scheme | wave: hand / loco distance | punch: hand / loco | shake: hand / loco | wave L-R cosine | punch L-R cosine | "
             "gesture centroid accuracy | gesture between / within |", "|---|---|---|---|---|---|---|---|"]
    for name in schemes:
        r = report[name]
        lines.append(f"| {name} | {r['wave']['hand_vs_loco_distance']:.2f} | {r['punch']['hand_vs_loco_distance']:.2f} | "
                     f"{r['shake']['hand_vs_loco_distance']:.2f} | {r['wave']['first_pair_cosine']:.3f} | "
                     f"{r['punch']['first_pair_cosine']:.3f} | {r['gesture_nearest_centroid_accuracy']:.0%} | "
                     f"{r['gesture_between_over_within']:.2f} |")
    table = "\n".join(lines)
    open(a.out + ".md", "w").write(json.dumps(check) + "\n\n" + table + "\n")
    print(json.dumps(check))
    print(table)


if __name__ == "__main__":
    main()
