# Evaluation protocol: all models, all 123 waving commands

`ALMI_trans/eval_protocol.py` on cir (CPU, 16 workers), 2026-10-14. Every model × the 123 distinct waving
captions of ALMI-X (41 locomotion commands × 3 hands) × 3 starts (seeds 0–2, initial joints ±0.05 rad),
8 s each: 369 runs per model. Per-run results: `runs.csv`.

| Model | Runs | Correct wave (95 % CI) | left / right / both | Walks in the commanded direction | Hand correct while upright | Upright 8 s | Mean survival (s) | DTW to commanded wave (median) |
|---|---|---|---|---|---|---|---|---|
| CL-20sl (paper) | 369 | 5 % (3–8 %) | 9 % / 0 % / 7 % | 95 % | 5 % | 100 % | 8.00 | 2.73 |
| CL-400sl (paper) | 369 | 0 % (0–1 %) | 0 % / 0 % / 0 % | 24 % | 22 % | 0 % | 2.04 | 1.51 |
| OL (paper) | 369 | 0 % (0–1 %) | 0 % / 0 % / 0 % | 5 % | 7 % | 0 % | 1.42 | 0.26 |
| A: text standardisation | 369 | 34 % (29–39 %) | 3 % / 0 % / 98 % | 99 % | 34 % | 100 % | 8.00 | 2.19 |
| **A + per-arm guidance (final)** | 369 | **82 % (78–86 %)** | 89 % / 61 % / 98 % | 95 % | 85 % | 94 % | 7.87 | 1.70 |
| C + guidance (mirror-trained) | 369 | 45 % (40–50 %) | 80 % / 32 % / 24 % | 96 % | 45 % | 100 % | 8.00 | 1.41 |

Measures:
* **Correct wave**: exactly the commanded arm(s) wave by the criterion of `wave_criterion.py` (hand at or above
  the shoulder, forearm swing ≥ 0.21 rad, for more than half of the 8 s; time after a fall counts as not
  waving). Validated against blind human labels: 22/24 on the final set, 78 % overall.
* **Hand correct while upright**: the same check over the time before a fall (needs ≥ 2 s upright), closer to
  the paper, which judges the hand even when the robot falls.
* **Walks in the commanded direction**: the paper's SR_low. Mean body-frame velocity while upright; every
  commanded axis has the commanded sign and ≥ 0.05 m/s or rad/s; "keep standing" needs all three < 0.1; a
  fall within 2 s fails.
* **DTW**: distance of the arms to the 5 nearest ALMI-X waves of the commanded hand (lower = closer to the
  demonstrations). A quality score, not a success test.

Findings:
* **The final method generalises to all 123 commands**: 82 % correct waves against 5 % for CL-20sl, with
  walking direction unchanged (95 %) and 94 % of runs upright for the full 8 s.
* **The paper's long-history models do not get both**: CL-400sl and OL fall within about 2 s and walk in
  the commanded direction in only 24 % and 5 % of runs; while still upright they wave the commanded hand in
  22 % and 7 %. OL's motion is closest to the demonstrations (DTW 0.26) because it replays decoded dataset
  motion open-loop. The paper's inference that long history is crucial for upper-body success is not
  needed: a 20-step model with better text conditioning and guidance reaches 82 %.
* **Right hand**: 61 % for the final method, the weakest hand but no longer near zero (CL-20sl 0 %).
* Mirror training plus guidance is better on the right hand than A alone but loses "wave both" (24 %).

Note: Mac and cir runs of the same model and command agree to 1e-7 rad at the start and then diverge
(chaotic contact dynamics amplify platform floating-point differences), so models are compared on one
machine, over many starts.
