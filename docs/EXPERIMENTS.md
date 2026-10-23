# Experiments

One entry per experiment: what we wanted to know, how we tested it, what came out, what we concluded.
Training and evaluation runs also have a row in [RUN_LOG.md](RUN_LOG.md); detailed write-ups and raw results
live under `results/<date>/`.

Template for new entries:

```
## E<n>. <short title> (<date>)
**Question / hypothesis.** ...
**Setup.** commit, machine, exact commands, data, number of runs.
**Results.** numbers (with intervals where relevant), table.
**Conclusion.** what this means, what it rules out, what to try next.
**Files.** results/<date>/...
```

## Index of earlier experiments (upper-body work, before this file existed)

| # | Date | Question | Outcome | Details |
|---|---|---|---|---|
| E1 | 2026-10-01 | Does CL-20sl use the hand word? | No: hand-swap ratio 1.00; CLIP hand variants 0.996 similar | `results/20261001/UPPER_BODY_DIAGNOSIS.md` |
| E2 | 2026-10-01/02 | Train A (text standardisation) and B (A + chunks of 10) | Offline hand sensitivity 1.57 (A), 3.52 (B) | RUN_LOG 2026-10-02 |
| E3 | 2026-10-10 | Can DTW to reference waves measure upper-body success? | 95 % on held-out ALMI-X, but only 6/9 agreement with human labels: rewards posture, not waving | `results/20261010/UPPER_BODY_DTW.md` |
| E4 | 2026-10-10 | Closed-loop comparison of baseline, A, B (human labels, 36 runs) | Baseline 1/12, A 5/12, B 2/12; temporal ensembling halves arm oscillation | same file |
| E5 | 2026-10-10 | Does executing chunks without averaging help B? | No: h ≥ 2 loses balance (h = 8: 1/72 upright), waving still below A | same file, `results/20261010/exec/` |
| E6 | 2026-10-10 | Hand guidance at run time (whole-body, s = 0.5–4) | Left hand fixed (24/24), "both" breaks; human: A + guidance 8/12 | same file, `results/20261010/guidance/` |
| E7 | 2026-10-10/11 | Why does "wave right" fail? Per-arm guidance, switching from a running both-wave, data balance | Not start-up, not text, not data quantity (28/30/33 clips); model's right-arm wave is weaker | same file |
| E8 | 2026-10-10/11 | C: mirror augmentation p = 0.5 | With guidance 59/72 by oscillation detector; humans: tie with A + guidance | `results/20261011/MIRROR.md` |
| E9 | 2026-10-11 | Explicit kinematic wave criterion (user's definition) | Agrees with blind labels 27/32; later 81 % over 104 runs (rater self-agreement 78 %) | `results/20261011/WAVE_CRITERION.md` |
| E10 | 2026-10-11 | C25: mirror augmentation p = 0.25 | 16–18/72: no gain | same file |
| E11 | 2026-10-11/12 | Stronger guidance; per-arm s = 12 | 61/72 on tuning runs; 47/60 (78 %) on held-out starts; humans 8/12 (right 0/4) | same file |
| E12 | 2026-10-14 | Evaluation protocol: 6 models × 123 commands × 3 starts | A + per-arm guidance 82 % (78–86 %), right 61 %; paper models 0–5 % | `results/20261014/PROTOCOL.md` |
| E13 | 2026-10-15 | H: hand token + condition dropout, cfg guidance | 8 % (cfg 3–11 %): stops waving | `results/20261015/MODEL_H.md` |
| E14 | 2026-10-16 | H0: hand token without dropout | 22 %, with guidance 44 % (A + guidance 86 % on the same runs) | same file |

---

## E15. Is the right-hand problem in the training data? (2026-10-23)

**Question / hypothesis.** The final method waves the right hand in only 61 % of protocol runs (left 89 %,
both 98 %), and most of its falls are right-hand commands. Hypothesis: the ALMI-X demonstrations of "wave
right" are fewer, smaller, or of lower quality than those of "wave left", so the model learns a weaker right
wave.

**Setup.** `ALMI_trans/analyse_wave_data.py` (commit 154c9bd + this script), on cir, single process, 39 s.
All 3,731 waving trajectories of the training set. Per trajectory: source clip, hand height above the
shoulder (MuJoCo forward kinematics of the recorded joint angles), shoulder pitch, forearm swing amplitude and
frequency, whether each arm meets the wave criterion (`wave_criterion.py`), and the robot's stability while
recording (torso tilt, base angular speed, leg joint speed). The waving arm of "wave left" is compared with
the waving arm of "wave right".

```
cd ALMI_trans && python analyse_wave_data.py --out ../results/20261023/wave_data
```

**Results** (median [25 %, 75 %]):

| Measure | "wave left": left arm | "wave right": right arm | "wave both": left / right arm |
|---|---|---|---|
| Trajectories / source clips | 1148 / 28 | 1230 / 30 | 1353 / 33 |
| Hand height above shoulder (m) | 0.035 [−0.029, 0.171] | 0.027 [−0.004, 0.141] | 0.032 / 0.086 |
| Share of time hand at/above shoulder | 0.58 [0.41, 0.66] | 0.57 [0.49, 0.80] | 0.53 / 0.62 |
| Shoulder pitch (rad) | −0.52 [−0.71, −0.36] | −0.53 [−1.00, −0.48] | −0.33 / −0.46 |
| Forearm swing (rad) | 0.35 [0.29, 0.46] | 0.49 [0.33, 0.53] | 0.34 / 0.32 |
| Swing frequency (Hz) | 0.86 [0.78, 0.98] | 0.93 [0.88, 1.04] | 0.68 / 0.69 |
| Waving arm meets the wave criterion | 74 % | 79 % | 58 % / 69 % |
| Resting arm meets the wave criterion | 0 % | 3 % | – |
| Torso tilt (projected gravity xy) | 0.053 [0.044, 0.065] | 0.060 [0.044, 0.078] | 0.080 |
| Base angular speed / leg joint speed | 0.058 / 0.041 | 0.055 / 0.040 | 0.055 / 0.039 |

**Conclusion.** The hypothesis is rejected. Right-hand demonstrations are as many as left-hand ones (1230 vs
1148 trajectories, 30 vs 28 source clips) and at least as good: they reach the same height, swing more (0.49
vs 0.35 rad) and slightly faster, and meet the wave criterion more often (79 % vs 74 %). The only difference
that could matter is that right waves are larger and tilt the torso a little more (0.060 vs 0.053), so they
are slightly more demanding to balance. The model sees enough good right-hand data; the weakness arises in
closed loop. Next hypotheses to test: (1) the MuJoCo robot model is not left-right symmetric (the mirror
replay of 2026-10-10 found 0.13–0.17 rad differences at the ankles), so right-arm waves disturb balance more
in MuJoCo than in the Isaac Gym simulator that generated the data; (2) the larger right-hand swing pushes the
closed-loop robot further from the training states, so guidance on the right arm makes it fall.

**Files.** `results/20261023/wave_data.csv` (one row per trajectory), `results/20261023/wave_data.md`,
`ALMI_trans/analyse_wave_data.py`.
