# Explicit wave criterion (forward kinematics)

Definition (from the human rater): *the hand raised to at least shoulder height, the forearm swinging side to
side, kept up for most of the 8 s; the swing must be large enough to be called a wave.* Implemented in
`ALMI_trans/wave_criterion.py`:

* hand height = middle-finger knuckle minus shoulder joint, in the torso frame (MuJoCo kinematics);
* swing = side-to-side angle of the forearm (elbow → hand) in the frontal plane, minus its 1 s moving average,
  RMS over 2 s windows;
* an arm waves when, in more than half of the 2 s windows of the test, the median height is ≥ 0 and the swing
  is ≥ SWING_MIN; time after a fall counts as not waving;
* SWING_MIN = half the median swing of the waving arm in ALMI-X single-hand waves = **0.210 rad**, set from the
  dataset only (`wave_criterion.json`). 83 % of demonstrated waving arms and 3 % of resting arms pass it.
  The reference waves keep the hand at or above the shoulder 70 % of the time.

**Check against the blind human labels (32 runs, not used for the setting): success agrees in 27 of 32 runs
(84 %)**, above the rater's own repeat agreement (25 of 32, 78 %). A threshold that only separates waving from
resting arms (0.067 rad) agreed in 23 of 32: small swings are not called waves by the rater.

## All models, 72 runs each (12 commands × 6 starts)

| Model | Correct wave | Left | Right | Both | Upright 8 s |
|---|---|---|---|---|---|
| Baseline CL-20sl | 8/72 (11 %) | 4/24 | 0/24 | 4/24 | 72/72 |
| A: text standardisation | 24/72 (33 %) | 0/24 | 1/24 | 23/24 | 72/72 |
| B: chunks, ensembling m = 0.5 | 9/72 (13 %) | 0/24 | 2/24 | 7/24 | 40/72 |
| B: chunks, no averaging (h = 1) | 5/72 (7 %) | 0/24 | 0/24 | 5/24 | 69/72 |
| A + guidance on single-hand commands | 33/72 (46 %) | 4/24 | 6/24 | 23/24 | 68/72 |
| C: mirror augmentation | 1/72 (1 %) | 0/24 | 0/24 | 1/24 | 72/72 |
| C + guidance s = 4 | 31/72 (43 %) | 23/24 | 8/24 | 0/24 | 72/72 |

Tie-break set (12 commands × 2 new starts per model, `results/videos/ub/tie/`), criterion: A + guidance 13/24
(left 4/8, right 2/8, both 7/8), C + guidance 10/24 (left 8/8, right 2/8, both 0/8). Human blind labels:
pending (`results/videos/ub/label_tie.html`).

## Tie-break, human blind labels (48 runs: 12 commands × 2 new starts × 2 models)

`human_labels_tie.csv`, criterion shown on the page, model hidden, random order.

| Model | Correct wave | Left | Right | Both |
|---|---|---|---|---|
| A + guidance on single-hand commands | 17/24 (71 %, 95 % CI 51–85 %) | 8/8 | 1/8 | 8/8 |
| C + guidance | 15/24 (63 %, 43–79 %) | 8/8 | 3/8 | 4/8 |

Paired by command and start: A + guidance alone succeeded in 5 pairs, C + guidance alone in 3 (sign test
p = 0.73). **No significant difference: the tie stands.** Over both blind sets (36 runs each): A + guidance
23/36 (64 %), C + guidance 21/36 (58 %). They differ in where they fail: A + guidance on "wave right", C +
guidance on "wave both".

Criterion against these labels: 35/48 (73 %); pooled over both blind sets 62/80 (78 %), equal to the rater's
own repeat agreement. Most misses are waves the rater accepted with a swing of 0.14–0.20 rad or held for 38–46 %
of the time, just under the 0.21 rad and 50 % limits.
