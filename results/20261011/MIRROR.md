# C: text standardisation + left-right mirror augmentation

Run `cl20sl_tn_mir_20261010_e6ce4c9wip` (cir, 20:15–01:47, 126k steps, final loss 0.00092 against 0.00063
for A; the mirrored windows make the data harder to fit). Hand-swap test, arm error swapped ÷ true: 1.28 at
20 steps (A 1.57).

MuJoCo, 12 commands × 6 starts, oscillation detector (78 % agreement with human labels):

| Model | Survive 8 s | Arm oscillation | Wave left: left | Wave right: right | Wave both: both | Total |
|---|---|---|---|---|---|---|
| A | 72/72 | 0.112 | 4/24 | 2/24 | 24/24 | 30/72 |
| A + guidance s = 4 | 68/72 | 0.070 | 24/24 | 11/24 | 4/24 | 39/72 |
| C | 72/72 | 0.064 | 9/24 | 0/24 | 1/24 | 10/72 |
| **C + guidance s = 4** | **72/72** | 0.085 | **24/24** | **18/24** | **17/24** | **59/72** |

* On its own, C waves less than A (mirroring averages each arm's behaviour over both sides).
* With guidance, C is the best and most uniform model: the right hand goes from 11/24 (A) to 18/24, "wave
  both" survives guidance (17/24, against 4/24 for A), and every run stays upright. One rule (s = 4) now
  works for all three hand commands.
* Human check pending: `results/videos/ub/label_cg4.html`.

Note: the first scoring of C picked up C_g4 files too (glob `C_*`); `eval_upper_body.sh` now scores
`<name>_go_*.npz` only. Earlier results are unaffected (no other name was a prefix of another when scored).

## Human check (strict: "none" unless the wave is done correctly)

`human_labels_cg4.csv`, C + guidance, deterministic start: left 3/4, right 1/4, both 1/4 → **5/12**. The
detector called 11 of these 12 correct, so under the stricter criterion it is far too lenient (an arm that
oscillates is not necessarily a correct wave). On these 12 runs the DTW distance to the commanded class
separates the judgements better (correct runs 1.09–1.39 except one 1.81; rejected runs 1.35–2.07), but 12 runs
are too few to set a threshold.

The earlier labels (baseline 1/12, A 5/12, A + guidance 8/12) used the looser "the arm moves back and forth"
criterion, so they are not comparable. A blinded re-labelling of A, A + guidance and C + guidance under the
strict criterion (32 runs, random order, model hidden): `results/videos/ub/label_blind.html`.

## Blind relabelling under the strict criterion (32 runs, random order, model hidden)

`human_labels_blind.csv`. Correct waves of the commanded hand:

| Model | Left | Right | Both | Total |
|---|---|---|---|---|
| A | 0/4 | 0/4 | 4/4 | 4/12 |
| A + guidance on single-hand commands (both = A) | 2/4 | 0/4 | 4/4 | 6/12 |
| C + guidance (one setting for all) | 3/4 | 2/4 | 1/4 | 6/12 |

* Rater consistency: the same runs got the same label as in the earlier sessions in 25 of 32 cases (78 %).
  With that noise and 12 runs per model, a difference of one or two runs is not meaningful; A + guidance and
  C + guidance are tied. C + guidance is the only model that waved the right hand alone (2 runs).
* Automatic measure: with these labels the oscillation detector agrees 75 %; refitting its threshold to
  0.09 reaches 88 % leave-one-out, but then almost no right-arm wave counts (the robot's right-arm
  oscillation is systematically smaller: the two right waves judged correct have 0.083–0.086), so a single
  threshold is biased against the right arm. Not adopted.
