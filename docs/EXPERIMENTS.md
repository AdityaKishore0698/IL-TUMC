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

## E16. Is the MuJoCo robot model left-right symmetric, and does it matter? (2026-10-23)

**Question / hypothesis.** E15 ruled out the data. Hypothesis: the MuJoCo H1-2 model used for evaluation is not
left-right symmetric, so right-arm waves disturb balance more than left-arm waves, which would explain the weak
right hand (61 %) and why most falls are right-hand commands.

**Setup.** Commit 154c9bd + `ALMI_trans/symmetry_test.py`, the symmetric model variants and the `<model>@<variant>`
option of `eval_protocol.py`; cir.
1. Static check (`python symmetry_test.py --out ../results/20261023/symmetry`, 49 s): every left body, joint,
   actuator and collision geom against its right counterpart.
2. Dynamic check without a policy (same script): robot standing on PD at the default pose while one arm replays an
   ALMI-X wave, own arm vs mirrored onto the other arm (40 waves).
3. Causal check: two corrected models, `scene_21dof_shin_none.xml` (no shin collision cylinder) and
   `scene_21dof_shin_both.xml` (on both shins), and the final method (A + per-arm guidance) on the released and
   both corrected models: `eval_protocol.py --models A+guidance A+guidance@shin_none A+guidance@shin_both --seeds 0 1 2 --jobs 16`
   (1,107 runs, about 35 min).

**Results.**
1. Masses, inertias, joint ranges, damping, armature, friction, actuator limits and PD gains are identical on both
   sides (largest differences: finger positions, ≤ 2 cm). **One real asymmetry:** the right shin (`right_knee_link`)
   has a collision cylinder (radius 0.02 m, half-length 0.1 m); the same cylinder on the left shin is commented out
   in the released `h1_2_21dof_no_left_leg.xml`.
2. Invalid: the robot falls in every run already on PD alone (the ankle gains cannot balance it without a policy),
   so this check says nothing about left vs right.
3. All three models give exactly the same scores:

| Robot model | Correct wave | Left / right / both | Upright 8 s |
|---|---|---|---|
| Released (right shin cylinder only) | 82 % (78–86 %) | 109 / **75** / 120 of 123 | 94 % |
| No shin cylinder | 82 % | 109 / **75** / 120 | 94 % |
| Shin cylinders on both legs | 82 % | 109 / **75** / 120 | 94 % |

   Of the 369 runs, 21 (no cylinder) and 20 (both) have different trajectories from the released model; these are
   runs in which the robot falls (22 falls in total), where the shins touch the ground after the fall. No upright
   run changes, and no success or survival outcome changes.

**Conclusion.** The released MuJoCo model does have a left-right asymmetry (only the right shin can collide), but it
only matters once the robot is already on the ground. It does not cause the right-hand weakness or the falls. Together
with E15 (data balanced and of equal quality), the right-hand weakness is in the learned policy's closed-loop
behaviour: in right-hand runs the robot falls 20 times out of 123 (left 2, both 0), so strong guidance on the right
arm destabilises it. Next: look at how the right-arm guidance changes the leg actions and the body's lateral balance
(for example guidance strength that is lowered when the torso tilts), or a smaller guidance strength for the right arm only.

**Files.** `results/20261023/symmetry.md`, `results/20261023/symmetry.json`, `results/20261023/symmetry_protocol_runs.csv`,
`ALMI_trans/symmetry_test.py`, `ALMI_trans/deploy/robots/h1_2/{h1_2_21dof,scene_21dof}_shin_{none,both}.xml`,
`ALMI_trans/deploy/deploy_mujoco/configs/h1_2_21dof_trans_shin_{none,both}.yaml`.

## E17. Does training-free per-arm guidance generalise beyond waving? (2026-10-23, finished 2026-10-24)

**Question / hypothesis.** Per-arm contrastive guidance raised correct waves from 5 % to 82 % (E12). Does the same
kind of guidance improve salute, punch and shake hand, or is it specific to waving? No transfer is assumed.

**Step 1: inspection (done).**
* **Blocker: every model so far was trained on waving only.** `dataset/dataset_CL_20sl.py` keeps only captions that
  contain "wave" (the paper trains on the waving subset), so CL-20sl, A, B, C, H and H0 never saw salute, punch or
  shake hand. A valid test needs a model trained on these gestures; guidance alone cannot produce a gesture the model
  was never trained on. Pilot with the waving model A on 12 gesture captions: salute 0/4, shake 0/4, punch 1/4
  (`results/20261023/gestures_pilot_A_runs.csv`); it waves instead.
* **ALMI-X data for the three gestures:**

| Gesture | Trajectories | Source clips | Caption phrases (× 41 locomotion commands) | Arms used |
|---|---|---|---|---|
| Salute | 1,640 | 2 | "salute" | right arm in every demonstration |
| Punch | 3,280 | 16 | "punch left" 1,435, "punch right" 1,435, "punch alternately with right and left hand" 205, "keep punching forward" 205 | named arm; both for the last two |
| Shake hand | 2,706 | 11 | "shake right hand" 2,214, "shake both hands" 492 (no "shake left hand") | named arm(s) |

* **Is the per-arm contrast meaningful for each gesture?** Each arm is contrasted with a caption in which only that
  arm's role changes (`ALMI_trans/gesture_guidance.py`):
  * punch left / right: like waving (left arm vs "punch right", right arm vs "punch alternately …" for "punch
    left"; mirrored for "punch right");
  * shake right hand: left arm vs "shake both hands"; the right arm has no contrast, because "shake left hand" does
    not exist;
  * salute: no contrast caption exists (one salute caption per locomotion command) → guidance is a no-op unless the
    exploratory fallback is used;
  * both-arm captions (punch alternately, keep punching forward, shake both hands, wave both): not guided, as for
    "wave both".
  The exploratory variant `--contrast_fallback loco` contrasts arms without a contrast caption with the
  locomotion-only caption ("Robot go backward slowly."), which never occurs in training.

**Step 2: model.** Model **G** = text standardisation (as A) trained on wave + salute + punch + shake (11,357
trajectories, 410 distinct captions), the paper's CL-20sl hyper-parameters, 126,000 steps as all earlier models.
`RUNS=tn_gest4 bash scripts/pipeline_upper_body.sh 20261023_9fed470wip` (new `--gestures` option), run
`cl20sl_tn_gest4_20261023_9fed470wip`, started 16:49 on cir.

**Step 3: success criteria (`ALMI_trans/gesture_criteria.py`), fixed from the demonstrations before any model is
evaluated** (`python gesture_criteria.py calibrate --n 300`, saved in `gesture_criteria.json`). Torso frame, relative
to each arm's shoulder joint, hand = middle-finger knuckle, 2 s windows, 8 s test, time after a fall counts as not
performing. A run succeeds when exactly the arms the caption asks for perform the gesture.
* Salute: hand within 0.15 m of the reference salute hand position (0.19 m forward, 0.04 m toward the midline,
  0.13 m above the shoulder) in more than half of the windows.
* Shake hand: hand held out in front (≥ 0.30 m forward) at waist-to-chest height (0.20–0.65 m below the shoulder)
  in more than half of the windows.
* Punch: at least one forward extension at shoulder height (hand ≥ 0.464 m forward while no more than 0.30 m below
  the shoulder; threshold that best separates punching from resting arms in the references).
* A first version (shake: within 0.15 m of the mean shake position; punch: two extensions) passed only 32 % and
  23 % of the demonstrated active arms and was replaced before evaluation.

Check on 300 demonstrations per gesture (active arms that pass / resting arms that pass / exact arm set):

| Criterion | Active arms | Resting arms | Exact |
|---|---|---|---|
| Salute | 100 % | 0 % | 100 % |
| Shake hand | 92 % | 0 % | 91 % |
| Punch | 100 % | 7 % | 94 % |

Specificity across gestures (share of arms on which a criterion fires, 150 demonstrations of each gesture): the punch
criterion also fires on 23 % of arms in wave demonstrations and 26 % in salute demonstrations (a raised arm reaches
forward at shoulder height). Punch success is therefore the least specific measure; human review of a sample of
punch runs is needed before claiming reliable punch success. Salute and shake criteria fire on ≤ 2 % of other
gestures.

**Planned evaluation (fixed before seeing results).** `ALMI_trans/eval_gestures.py`, chained to start when G is
exported (`~/ALMI/bin/chain_gestures.sh` on cir):
1. Main: every caption of salute (41), punch (164), shake (82) and wave (123) × starts 0, 1, 2, for G, G + gesture
   guidance (s = 12) and G + gesture guidance + locomotion fallback: 3 × 410 × 3 = 3,690 runs.
2. Held-out starts 5, 6 for salute, punch, shake, G and G + gesture: 2 × 287 × 2 = 1,148 runs.
Reported per gesture: success with 95 % Wilson intervals, success by expected arm(s), success while upright,
not-performed-while-upright, falls, walking direction, 8 s survival, run time per 8 s run (guidance overhead).

**Runs completed.** G trained 16:49–22:43 (final loss 0.00062), exported 22:45. Main evaluation 3,690 runs
(22:45–23:54), held-out starts 1,148 runs (23:54–00:15), no failed runs. Raw results:
`results/20261023/gestures/{main,heldout}_{runs.csv,summary.md}`.

**Results** (G = four-gesture model; guidance = per-arm contrastive guidance, s = 12; successes / runs, 95 % Wilson
interval; "success and upright" = gesture performed and the robot still upright at 8 s):

| Gesture | Starts | Condition | Gesture success | Success and upright 8 s | By expected arm(s) | Falls | Walks as commanded | Paired G → guided: gains / losses (sign test) |
|---|---|---|---|---|---|---|---|---|
| Salute | 0–2 | G | 113/123 = 92 % (86–96 %) | 92 % | right 113/123 | 0 | 116/123 | |
| | | G + guidance | 113/123 = 92 % | 92 % | identical (no contrast caption exists) | 0 | 116/123 | 0 / 0 |
| | 5–6 | G / G + guidance | 76/82 = 93 % / 93 % | 93 % / 93 % | | 0 / 0 | 80/82 | 0 / 0 |
| Punch | 0–2 | G | 251/492 = 51 % (47–55 %) | 251/492 = 51 % | left 71/123, right 21/123, both 159/246 | 0 | 427/492 | |
| | | G + guidance | 292/492 = 59 % (55–64 %) | **205/492 = 42 % (37–46 %)** | left 97/123, right 36/123, both 159/246 | **194** | 353/492 | 64 / 23 (p < 0.001) |
| | 5–6 | G | 165/328 = 50 % (45–56 %) | 50 % | left 40/82, right 13/82, both 112/164 | 0 | 286/328 | |
| | | G + guidance | 206/328 = 63 % (57–68 %) | **140/328 = 43 % (37–48 %)** | left 58/82, right 36/82 | **134** | 237/328 | 54 / 13 (p < 0.001) |
| Shake hand | 0–2 | G | 145/246 = 59 % (53–65 %) | 59 % | right 123/123, both 22/123 | 0 | 237/246 | |
| | | G + guidance | 137/246 = 56 % (49–62 %) | 56 % | right 115/123, both 22/123 | 0 | 227/246 | 0 / 8 (p = 0.008) |
| | 5–6 | G / G + guidance | 99/164 = 60 % / 92/164 = 56 % | 60 % / 56 % | right 82/82 → 75/82 | 0 / 0 | 160 / 154 of 164 | 0 / 7 (p = 0.016) |
| Wave | 0–2 | G | 19/369 = 5 % (3–8 %) | 5 % | left 3, right 15, both 1 of 123 | 0 | 364/369 | |
| | | G + guidance | 121/369 = 33 % (28–38 %) | 108/369 = 29 % | left 107, right 13, both 1 of 123 | 23 | 337/369 | 116 / 14 |
| | | *(A + guidance, E12, waving-only model)* | *82 %* | | *109 / 75 / 120 of 123* | | | |

Exploratory locomotion-caption fallback (`G+gesture+loco`): salute 9/123 = 7 % with 115 falls, shake 25/246 = 10 %
with 115 falls; punch and wave identical to G + guidance (no fallback used). Contrasting with a caption the model
never saw drives the arms into states that topple the robot. Run time per 8 s run (wall clock, one CPU core): 16.4 s
without guidance, 17.1–20.6 s with guidance (one or two extra forward passes per step).

**Failure modes.**
* Punch, guided: guidance makes the commanded arm punch more often (left 71 → 97, right 21 → 36 of 123) but the
  robot falls in 39 % of punch runs (194/492, none without guidance), mostly after the punch, so success with the
  robot still upright drops from 51 % to 42 %. The extra punch success comes with a stability cost that outweighs it.
  Without guidance, "punch right" is mostly answered with the left arm (52/123) or both (32/123): the same
  right-side weakness as for waving, although the data have as many right as left punches (1,435 each).
* Shake hand, guided: only the left arm can be guided ("shake both hands" as contrast); this costs 8 right-hand
  successes (right arm no longer held out) and gains nothing. "Shake both hands" is mostly performed with the right
  hand only (79/123), both without and with guidance (not guided).
* Salute: G already salutes in 92 % of runs; there is no contrast caption, so guidance cannot be applied.
* Wave: G waves much worse than A (5 % vs 34 % without guidance, 33 % vs 82 % with). The hand-swap test of G gives
  an arm-action change of 0.0048 when the hand word is swapped (A: 0.0103, the paper's CL-20sl: 0.0038): with the
  caption statistics taken over 410 captions of four gestures, the gesture words dominate the variance and the
  standardisation no longer amplifies the hand words. Training on more gestures undid most of the benefit of text
  standardisation for hand selection.

**Conclusion.** Training-free per-arm contrastive guidance does **not** generalise as a free improvement. It
needs a contrast caption that flips exactly one arm's role; ALMI-X provides one only for punch (and waving). For
punch it raises gesture success (51 → 59 %, held out 50 → 63 %; paired p < 0.001) but makes the robot fall in about
40 % of runs, so success with the robot upright drops (51 → 42 %, held out 50 → 43 %). For shake hand the only
available contrast slightly hurts (59 → 56 %); for salute no contrast exists, and the exploratory locomotion-only
contrast makes the robot fall. Separately, the four-gesture model is good at salute (92 %) and right-hand shake
(100 %) without guidance, but weak at waving and at "punch right", and the text standardisation that made A
hand-selective is diluted when the caption set grows.

**Limitations.** Punch success uses the least specific criterion (it also fires on 23–26 % of arms in wave and
salute demonstrations) and has not been reviewed by eye; no gesture criterion in E17 has human validation yet.

**Next experiments justified by these results.**
1. Human review of a sample of punch and shake runs (G and G + guidance) to validate the two criteria.
2. Punch guidance strength sweep (s = 2, 4, 8) with "success and upright" as the target, to find whether a weaker
   push keeps the gain without the falls (as per-arm guidance needed s = 12 for waving but whole-body guidance fell
   above s = 4).
3. Text standardisation with statistics per gesture family (or over the hand-variant captions only), to restore
   hand selectivity in the four-gesture model.
4. The right-side weakness now appears for two gestures (wave and punch) with balanced data: compare left and right
   arm action magnitudes and the robot's lateral balance in G's closed-loop runs.

**Files.** `ALMI_trans/{gesture_features,gesture_criteria,gesture_guidance,eval_gestures}.py`,
`ALMI_trans/gesture_criteria.json`, `results/20261023/gesture_data.csv`, `results/20261023/gestures_pilot_A_runs.csv`.

## E18. Weaker punch guidance: can it keep the gain without the falls? (continuation of E17, 2026-10-26, finished 19:19)

**Question / hypothesis.** In E17, per-arm guidance at s = 12 raised punch success with model G (51 → 59 %) but the
robot fell in 194 of 492 runs, so successful punches with the robot still upright at 8 s dropped (251 → 205). For
waving, whole-body guidance fell above s = 4 while per-arm guidance tolerated s = 12. Hypothesis: a weaker strength
(s = 2, 4 or 8) keeps part of the punch gain while preserving balance.

**Verification of E17 (from `results/20261023/gestures/{main,heldout}_runs.csv`).** All recorded punch numbers agree
with the raw results: main G 251/492 successes, 0 falls; G + guidance (s = 12) 292/492, 194 falls, 298 upright at
8 s, 205 successes with the robot upright at 8 s (87 successful punches were followed by a fall); held out G 165/328,
0 falls; s = 12 206/328, 134 falls, 140 successes and upright. Scoring: a punch counts if the hand reaches 0.464 m
forward at shoulder height at least once before any fall; a run succeeds when exactly the requested arm(s) punch.
Guidance strength is the `--hand_guidance` value passed by `eval_gestures.py` (s = 12 for `G+gesture`).

**Setup.** Commit 34ab41f + `eval_gestures.py` model names `G+gesture<s>` (only change). Model G
(`policy_trans_cl20sl_tn_gest4_20261023_9fed470wip.pt`), not retrained. All 164 punch captions (41 locomotion
commands × "punch left", "punch right", "punch alternately with right and left hand", "keep punching forward"),
starts 0–2 (main) and 5–6 (held out), success criteria unchanged (`gesture_criteria.json`). The G and s = 12 runs of
E17 are reused (same checkpoint, code verified identical on cir by checksum, same captions and starts), copied into
new folders; s = 2, 4, 8 are new. Guidance applies only to "punch left" and "punch right"; the two both-arm captions
are not guided at any strength. Pilot (4 captions, start 0, s = 2) ran and scored correctly. cir was shared with
other users' jobs (load about 35), so run times are not comparable with E17's.

```
bash ~/ALMI/bin/run_e18.sh     # on cir: eval_gestures.py --gestures punch --models G G+gesture2 G+gesture4 \
                               #   G+gesture8 G+gesture --seeds 0 1 2 (then 5 6) --jobs 12 --out output/gestures_e18[_heldout]
```

Trials: 492 runs per strength (main) + 328 (held out); 1,476 + 984 new runs. All new runs completed (0 failed);
main 17:09–18:30, held out 18:30–19:19. Raw results: `results/20261026/e18_{main,heldout}_{runs.csv,summary.md}`.

**Results.** "Success and upright" = the requested arm(s) punched and the robot was still standing at 8 s (the main
outcome). Paired: same caption and start as no guidance; gains / losses of "success and upright" and sign test.

Main (starts 0–2, 492 runs each):

| Guidance | Punch success | **Success and upright** (95 % CI) | Single-arm captions: success and upright | Falls | Upright 8 s | Walks as commanded | Paired vs none: gains / losses (p) |
|---|---|---|---|---|---|---|---|
| none (G) | 251 = 51 % | **251 = 51 % (47–55 %)** | 92/246 | 0 | 492 | 427 | – |
| s = 2 | 210 = 43 % | 210 = 43 % (38–47 %) | 51/246 | 5 | 487 | 420 | 32 / 73 (p < 0.001) |
| s = 4 | 178 = 36 % | 174 = 35 % (31–40 %) | 15/246 | 46 | 446 | 394 | 9 / 86 (p < 0.001) |
| s = 8 | 261 = 53 % | 232 = 47 % (43–52 %) | 73/246 | 125 | 367 | 361 | 28 / 47 (p = 0.04) |
| s = 12 (E17) | 292 = 59 % | 205 = 42 % (37–46 %) | 46/246 | 194 | 298 | 353 | 24 / 70 (p < 0.001) |

Held out (starts 5–6, 328 runs each):

| Guidance | Punch success | **Success and upright** (95 % CI) | Single-arm: success and upright | Falls | Upright 8 s | Walks as commanded | Paired vs none |
|---|---|---|---|---|---|---|---|
| none (G) | 165 = 50 % | **165 = 50 % (45–56 %)** | 53/164 | 0 | 328 | 286 | – |
| s = 2 | 153 = 47 % | 153 = 47 % (41–52 %) | 41/164 | 1 | 327 | 283 | 21 / 33 (p = 0.13) |
| s = 4 | 134 = 41 % | 134 = 41 % (36–46 %) | 22/164 | 20 | 308 | 271 | 15 / 46 (p < 0.001) |
| s = 8 | 182 = 55 % | 166 = 51 % (45–56 %) | 54/164 | 81 | 247 | 241 | 26 / 25 (p = 1.0) |
| s = 12 | 206 = 63 % | 140 = 43 % (37–48 %) | 28/164 | 134 | 194 | 237 | 16 / 41 (p = 0.001) |

The two both-arm captions are never guided and score the same at every strength (159/246 main, 112/164 held out).

**What the requested arm did** (main, "punch left" / "punch right", 123 each; arms that punched):

| Guidance | "punch left" → left / both / other | "punch right" → right / both / left / none |
|---|---|---|
| none | 71 / 48 / 4 | 21 / 32 / 52 / 18 |
| s = 2 | 10 / 113 / 0 | 41 / 48 / 21 / 13 |
| s = 4 | 2 / 121 / 0 | 17 / 46 / 12 / 48 |
| s = 8 | 76 / 46 / 1 | 26 / 65 / 14 / 18 |
| s = 12 | 97 / 26 / 0 | 36 / 80 / 3 / 4 |

**Run time.** Not comparable between strengths: cir was shared with other users' jobs at load 30–40, and the median
time per 8 s run varied from 21.8 to 64.8 s without a pattern in the strength. The unloaded E17 measurement remains
the reference: 16.4 s without guidance, 18.7 s with guidance (two extra forward passes per control step).

**Conclusion.** The hypothesis is not supported. No guidance strength improves successful punches with the robot
still upright over no guidance. Weak guidance is worse than none: at s = 2 and 4, "punch left" turns into punching
with both arms (113 and 121 of 123), so success drops while the robot stays mostly upright. Stronger guidance (s = 8,
12) recovers and exceeds the raw punch success but the robot falls in 25–39 % of runs. s = 8 is the best guided
setting (main 47 %, held out 51 % success and upright) and is not better than no guidance (51 % and 50 %; held-out
paired gains 26, losses 25). Falls grow steadily with strength (5, 46, 125, 194 of 492). The non-monotonic effect
on "punch left" suggests the right-arm contrast caption ("punch alternately with right and left hand") is a poor
contrast: its right-arm prediction depends on where the robot is in the alternation, so the push it produces on the
right arm can drive it into punching. This is a hypothesis, not tested here.

**Limitations.** The punch criterion is the least specific gesture criterion (E17) and has no human validation;
"success" counts one forward extension at shoulder height. One model checkpoint; guidance applies only to the
single-arm captions (246 of 492 runs); run times were measured on a shared machine.

## E19. Does standardising text across four gestures dilute left/right? (2026-10-26, step 1: diagnostic, no training)

**Question / hypothesis.** Model G (four gestures) uses the hand word much less than model A (waving only): hand-swap
arm change 0.0048 vs 0.0103 (paper's CL-20sl 0.0038), correct waves 5 % vs 34 % without guidance. Hypothesis: the
text standardisation computed over the captions of four gesture families makes left/right a smaller part of the
standardised embedding than standardisation over waving captions alone.

**Inspection.**
* Standardisation (`train_almi_cl_20sl.py`, `--text-norm`): the mean and standard deviation (unbiased, clamped at
  1e-3) of the CLIP ViT-B/32 embedding over the *distinct* training captions, CLIP on the GPU with fp16 weights;
  stored in the checkpoint (`trans_base.text_mean/std`) and applied inside the model, so the exported policy applies
  it too. Model A: 123 waving captions; model G: 410 captions (wave 123, punch 164, shake 82, salute 41). Both use
  seed 123, batch 128, lr 1e-4, 126k steps; they differ in the training data (3,731 vs 11,357 trajectories) and the
  standardisation statistics. G therefore also sees each waving trajectory about one third as often as A.
* Hand-swap test (`eval_text_sensitivity.py`): 2,000 windows of the *waving* training data (the dataset loader's
  default), arm-action change and arm-error ratio when the hand word is swapped. Same for A and G.

**Diagnostic** (`ALMI_trans/text_norm_diagnostic.py`, cir, 1 min): all 410 captions embedded exactly as in training,
standardised five ways: none (raw); A's stored statistics; G's stored statistics (recomputed here to within 6e-8, so
the reconstruction is exact); per-family statistics; and hierarchical (family statistics for the within-family part
plus G's statistics for the family means, so families stay apart). Measures on the standardised embeddings: distance
between captions differing only in the hand phrase divided by the median distance between captions differing only in
the locomotion command (hand / loco), cosine of the left/right pair, and whether gestures stay separable
(leave-one-out nearest family centroid; between- / within-family distance).

| Scheme | wave hand / loco | punch hand / loco | shake hand / loco | wave L–R cosine | punch L–R cosine | gesture centroid accuracy | between / within |
|---|---|---|---|---|---|---|---|
| raw (paper) | 0.32 | 0.76 | 0.48 | 0.996 | 0.995 | 100 % | 1.38 |
| A's statistics | 0.33 | 0.83 | 0.54 | 0.854 | 0.951 | 100 % | 1.59 |
| G's statistics | 0.32 | 0.76 | 0.52 | 0.914 | 0.909 | 100 % | 1.36 |
| per family | 0.33 | 0.74 | 0.51 | 0.854 | 0.873 | **0 %** | 0.00 |
| hierarchical | 0.33 | 0.74 | 0.51 | 0.898 | 0.897 | 100 % | 1.09 |

**Findings.**
1. **The hypothesis is not supported at the input.** Relative to the locomotion part of the command, the hand phrase
   is equally visible under every scheme (wave 0.32–0.33). G's statistics do not shrink it compared with A's. Only
   the cosine changes, because centring moves the origin; the model's first layer is linear with a bias, so a shift
   of the origin carries no information.
2. **Per-family standardisation as proposed is invalid**: centring each family separately removes what distinguishes
   the gestures (family centroid accuracy 0 %), so the model could no longer tell "wave left" from "punch left" from
   the text. The hierarchical variant fixes that but leaves the hand signal where it was (0.33).
3. Per-dimension standardisation is an affine change of the input to a linear layer: it cannot add or remove
   information, only change how easy it is to learn. A's gain over the paper's model (hand-swap 0.0038 → 0.0103) must
   therefore come from training dynamics, and the A–G difference can come from the statistics *or* from the training
   data (G sees waving a third as often and must also learn three other gestures). The diagnostic cannot separate
   these; a controlled training run can.

**Proposed controlled experiment (not started).** One run, **A-Gstats**: model A's setup exactly (waving data only,
same seed, batch, learning rate, 126k steps), but with G's standardisation statistics instead of A's. A and A-Gstats
then differ only in the statistics; G and A-Gstats differ only in the training data. Compare hand-swap sensitivity,
correct waves with and without guidance (wave protocol, 369 runs each) and falls.
* If A-Gstats ≈ A: the statistics are not the cause; the loss comes from multi-gesture training (data mix / exposure).
* If A-Gstats ≈ G: the statistics are the cause.
Files to change: `ALMI_trans/options/option_transformer.py` and `train_almi_cl_20sl.py` (a `--text-norm-from <checkpoint>`
option that copies the statistics instead of computing them), `scripts/pipeline_upper_body.sh` (variant), and
`ALMI_trans/eval_protocol.py` (model entry). About 6 h of training and 1 h of evaluation on cir.

**Files.** `ALMI_trans/text_norm_diagnostic.py`, `results/20261026/text_norm_diagnostic.{md,json}`.

### E19, step 2: controlled run A-Gstats (started 2026-10-26 22:04)

**Setup verified before launch.** Smoke test (100 steps): the training log lists the same 3,731 waving trajectories
as A, and the saved checkpoint's `text_mean` / `text_std` are bit-identical to G's. The argument lists of A and the
smoke run differ only in run names, step count of the smoke test, options added after A whose defaults reproduce A's
behaviour (gestures = wave, mirror_aug = 0, hand_token = False, hand_drop = 0, text_drop = 0) and the new
`--text-norm-from`. Same seed (123), batch 128, learning rate 1e-4, architecture, 126,000 steps. Caveat: the data
loader shuffles with Python's unseeded `random`, so A itself is not bit-reproducible; A and A-Gstats see the same data
in a different order. Inference uses the statistics stored in the checkpoint and exported with the policy, so
training and evaluation use G's statistics consistently.

Run `cl20sl_tn_Gstats_20261026_34ab41fwip` (`RUNS=tn_Gstats bash scripts/pipeline_upper_body.sh 20261026_34ab41fwip`),
new files only: `output/cl20sl_tn_Gstats_20261026_34ab41fwip/`, `output/export/policy_trans_cl20sl_tn_Gstats_20261026_34ab41fwip.pt`.
After export the pipeline runs the hand-swap test; `~/ALMI/bin/chain_e19.sh` then runs the waving protocol
(123 captions × starts 0–2) for A-Gstats and G with and without per-arm guidance (s = 12) into `output/protocol_e19`,
where A's E12 logs are reused (same code and machine; E16 reproduced them exactly).

**Results.** Pending.
