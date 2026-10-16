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

## H0: the hand token without dropout

Run `cl20sl_tn_hand0_20261015_3dfe209wip` (cir, 11:52–20:33, 126k steps, final loss 0.00064). Hand-swap test,
arm error swapped ÷ true at 20 steps: **1.91–2.10**, the strongest hand sensitivity of any one-action model
(A 1.57, H 1.40–1.50).

Same 12-command grid × 3 starts, run together with A on cir (`h0grid_runs.csv`):

| Model | Correct wave (95 % CI) | left / right / both | Upright 8 s |
|---|---|---|---|
| A | 33 % (20–50 %) | 0 / 0 / 100 % | 100 % |
| H0 | 22 % (12–38 %) | 17 / 8 / 42 % | 100 % |
| **A + per-arm guidance** | **86 % (71–94 %)** | 92 / 67 / 100 % | 92 % |
| H0 + per-arm guidance | 44 % (30–60 %) | 92 / 0 / 42 % | 92 % |

* The hand token removes the dropout problem (H0 waves; H did not) and is the only change that gives some
  single-hand waves without guidance (left 17 %, right 8 %, against 0 % for A).
* But it loses "wave both" (42 % against 100 %), and with guidance it is far behind A (44 % against 86 %);
  its right hand under guidance is 0 %.
* Offline hand sensitivity (2.0) does not translate into closed-loop waving.

Conclusion for step 2: an explicit hand token does not improve on text standardisation plus per-arm guidance.
Fixing the CLIP features was sufficient for the model to receive the hand; the remaining gains come from how
the hand is used at run time (guidance). Not taken to the full protocol.
