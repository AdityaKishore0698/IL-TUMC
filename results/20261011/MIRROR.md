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
