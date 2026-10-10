# Objective upper-body measure (DTW) and first closed-loop results

ALMI judges upper-body success by watching whether the robot waves the correct hand (15 trials in its
Table 13). `ALMI_trans/upper_body_metric.py` replaces that with dynamic time warping (DTW) against the
ALMI-X reference motions.

## Measure

* Each arm's executed joint trajectory (4 joints, offsets from the default pose, 50 Hz) is compared with
  the same arm in 60 reference trajectories per hand class (wave left / right / both, one per distinct
  motion) from the 3,731 waving trajectories of ALMI-X.
* DTW with a 1 s band, Euclidean step cost, divided by the length. Both sequences are cut to the shorter
  length; only the time before a fall is used (at least 1 s, otherwise the run is unscorable).
* Score of a class = mean of its 5 nearest references (left arm + right arm). The run waves the nearest
  class; if even that is farther than 2.0 (the worst held-out ALMI-X wave), it is "no clear wave".
* Success = the class equals the commanded hand. The distance to the commanded class is a quality score.

## Validation

**On ALMI-X, held-out motions** (motions never used as references):

| Window | Accuracy | Confusion |
|---|---|---|
| 400 steps (8 s) | 285/300 = 95.0 % | 15 "left" clips classed "both"; right and both 100 % |
| 100 steps (2 s) | 172/180 = 95.6 % | 8 "left" clips classed "both" |

Held-out waves score 0.29 (median) to the nearest class, 0.70 at the 90th percentile, 2.04 at most.

**Against the visual judgements** (the 9 Table 13 runs regenerated with logging, deterministic):

| Run | Judged by eye | DTW measure | Distance to nearest class |
|---|---|---|---|
| OL, wave left / right / both | left / right / both | left / right / both | 0.14 / 0.41 / 0.17 |
| CL-400sl, wave right | right | right | 0.47 |
| CL-400sl, wave both | both | both | 1.78 |
| CL-400sl, wave left | unclear (fell first) | right, small margin | 1.73 |
| CL-20sl, wave left / right / both | left / both / left | no clear wave (×3) | 2.14 / 2.17 / 2.27 |

The measure agrees with every clear judgement of CL-400sl and OL. For CL-20sl it reports no recognisable
wave: in all three commands CL-20sl moves both arms by the same amount (activity 0.26 / 0.27 rad, the
same for every command) and is 3× farther from any reference wave than real waves are. This matches the
earlier finding that CL-20sl ignores the hand word.

## Closed-loop comparison (12 commands: 4 locomotion × 3 hands, 8 s, MuJoCo)

| Model | Upper-body success | No clear wave | Mean DTW to asked class | Mean survival |
|---|---|---|---|---|
| Baseline CL-20sl | 0/12 | 11 | 2.68 | 8.00 s |
| A: text standardisation | 2/12 | 10 | 2.16 | 8.00 s |
| B: A + chunks of 10, ensemble m = 0 (uniform) | 7/12 | 0 | 1.76 | 4.03 s |
| B, m = 0.1 | 7/12 | 0 | 1.76 | 4.54 s |
| B, m = 0.3 | 8/12 | 0 | 1.49 | 6.69 s |
| **B, m = 0.5** | **8/12** | 0 | 1.65 | **7.47 s** (10 of 12 survive 8 s) |
| B, m = 1 | 4/12 | 0 | 2.30 | 7.43 s |
| B, m = 2 | 5/12 | 0 | 2.16 | 8.00 s |
| B, m = 50 (newest chunk only, no ensembling) | 4/12 | 0 | 2.29 | 7.09 s |

m weights the prediction made i steps ago by exp(−m·i): small m averages over the whole chunk (smooth,
committed arm motion), large m follows the newest prediction (reactive balance).

* Action chunking is what makes the arms follow the text: B waves a recognisable, correct wave in up to
  8 of 12 commands, against 0 for the baseline and 2 for text standardisation alone.
* Temporal ensembling sets the balance-versus-arm trade-off; m = 0.5 keeps most of the arm gain while
  10 of 12 runs survive the full 8 s. Without ensembling (m = 50) both arms and balance are worse.
* Remaining failure: at m = 0.5, "wave left" 4/4 and "wave both" 4/4, but "wave right" 0/4 (B waves both
  arms). The two falls are both "go left fast".

Caveat: MuJoCo is deterministic, so each command is one run; differences of one or two commands are
within what a different command set could change.
