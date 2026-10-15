# H: explicit hand token + condition dropout

Run `cl20sl_tn_hand_20261014_629b6d1wip` (cir, 20:40–05:20, 126k steps, final loss 0.00065, same as A).
Hand-swap test (caption and token swapped together), arm error swapped ÷ true at 20 steps: 1.40–1.50 (A 1.57).

Guidance sweep, 12-command grid × 3 starts (36 runs each), `eval_protocol.py` on cir:

| Model | Correct wave (95 % CI) | left / right / both | Upright 8 s | DTW (median) |
|---|---|---|---|---|
| H | 8 % (3–22 %) | 0 / 0 / 25 % | 100 % | 1.71 |
| H + cfg s = 1 / 2 / 4 / 8 | 6 / 11 / 6 / 3 % | left 0 %, right 0 % | 100 % | 1.54–1.66 |
| H + per-arm s = 12 | 31 % (18–47 %) | 67 / 0 / 25 % | 97 % | 1.36 |
| (A + per-arm s = 12, full protocol) | 82 % | 89 / 61 / 98 % | 94 % | 1.70 |

H mostly does not wave at all: on "wave left" no run waves; the median fraction of time an arm meets the
wave criterion is 0–15 %. Classifier-free guidance against the hidden hand token does not restore it. H is
worse than A on every hand.

Likely cause: the dropout. With the caption hidden in 10 % and the hand hidden in 15 % of windows, the model
also learns to predict arm actions without knowing which arm waves, and the average of left, right and both
waves is an arm that hardly moves; in closed loop that low-amplitude behaviour takes over. The contrast with
the hidden hand then mostly says *which* arm, not *whether* to wave. Separating the two needs H0: the hand
token without dropout.
