# Experiment map and training recipe

This is the short technical companion to [COLLEAGUE_OVERVIEW.md](COLLEAGUE_OVERVIEW.md).
Counts are simulator successes. Different experiment rows use different
frozen seeds, so compare methods **within a row**, not raw counts across rows.
Full configs, logs, Wilson intervals, paired fixed/broken counts, and exact
McNemar tests are in `docs/residual/NOTES.md` and `docs/residual/results/`
on the `residual-rl` branch.

## Base policy and observation

| Task ID | Frozen SmolVLA model | Training step |
| --- | --- | ---: |
| `green_push_forward` | `tsangb34/smolvla_tri_green_push_100`, `checkpoints/last` | 80000 |
| `cylinder_right_of_cube` | `tsangb34/smolvla_tri_cylinder_right_100`, top-level weights | 80000 |
| `cube_flip` | `tsangb34/smolvla_tri_cube_flip_100`, `checkpoints/100000` | 100000 |

SmolVLA supplies 50 open-loop absolute targets per inference. We apply
a correction to each 30 Hz target. Joint order is shoulder pan, shoulder
lift, elbow flex, wrist flex, wrist roll, gripper; arm units are degrees
and the gripper uses 0–100. The deployed constant-offset DAgger clips
each arm correction to ±4° and never changes the gripper.

The 31 no-leak features per step are six measured joints, six current
SmolVLA targets, six previous commanded targets, six previous-command
minus measured-joint values, one position within the 50-action chunk,
and six previous residual values. Joint quantities are normalized by
joint-limit midpoints and half-ranges. Ten features vectors are stacked
oldest to newest for the deployed model, with the first vector repeated
at episode start. No simulator pose, true error, object state, or contact
state is an online model input.

## Training and evaluation

**Residual PPO:** a separate 2×256 actor and critic per task, 500,000
environment steps, eight parallel environments, learning rate
`1e-4`, target KL `0.02`. The task reward gave incremental object
progress plus `+1` on success; the dense-reward variant additionally
penalized residual-plus-true-offset using simulator ground truth in the
**reward only**. The action started at zero mean. Frozen SmolVLA never
received a gradient. Cube PPO helped; green and cylinder task-reward
PPO and green dense-reward PPO did not.

**Oracle and supervised probe:** in separate training episodes at ±2°
constant offsets, the simulator supplied the true five-joint offset as
a label. An oracle used its negative as correction on 50 selection
episodes. Two 2×256 Tanh MLPs learned the offset from one 31-feature
step or ten stacked steps. Eighty episodes trained them; 20 separate
episodes assessed held-out error. Despite good offline prediction,
the models performed badly online. A check replaced the online
prediction with the true offset in the *same* control path and recovered
the oracle's episode outcomes, confirming the sign, units, and joint
order.

**Constant-offset DAgger:** begin with those 80 training episodes.
For each of three rounds, collect 50 additional training-seed rollouts
using the latest predictor's corrections; 15/50 use a random,
episode-fixed arm correction in ±3°. Fit both 1-step and 10-step
2×256 Tanh offset predictors to **all** accumulated steps after each
round (Adam `1e-3`, 20 epochs, batch 1024, normalized inputs).
The true offset is a training label only. Evaluate each round on 50
selection episodes. Before looking at test data, lock the round-3
10-step model file by SHA256. Its online command is
`clip(base - predicted_offset, ±4° correction and joint limits)`.
Each task has its own model.

**Evaluation discipline:** training, selection, 100-episode test,
200-episode confirmation, and 50-episode zero-error sets are disjoint.
Within a set, methods use the same starts and random seeds. Report
success, Wilson 95% interval, fixed/broken against SmolVLA, and exact
paired McNemar p. The constant-offset DAgger test/confirmation results
are in [COLLEAGUE_OVERVIEW.md](COLLEAGUE_OVERVIEW.md).

**Classical comparisons:** an integral estimate of previous command
minus measured joint was tuned over gains `0.01, 0.03, 0.1, 0.3`.
A constant-offset Kalman filter was tuned over process noise
`1e-6, 1e-4` × measurement noise `0.1, 1.0 deg²`, with every-step
or low-velocity-only updates: eight settings. Settings were chosen on
50 selection episodes and locked before tests. Under constant offsets,
the best integral and Kalman settings generally matched or exceeded
DAgger.

## Actuator-error follow-ups

Each cell below is **SmolVLA / 10-step DAgger / tuned integral / tuned
constant-offset Kalman**, successes out of 100 on paired frozen starts.
The DAgger in each stress row was trained *for that row's error model*;
these are separate models from the three deployed constant-offset ones.

| Error model | Green push | Cylinder | Cube flip |
| --- | --- | --- | --- |
| Constant ±2° offset, classical comparison rerun | 71/98/100/100 | 65/98/94/100 | 61/100/100/100 |
| Backlash + gain error, pre-registered | 66/97/95/97 | 64/96/77/93 | 60/100/97/96 |
| Offset jumps | 57/100/96/97 | 54/99/78/87 | 59/100/95/100 |
| Cyclic position error | 89/98/100/98 | 84/97/83/88 | 81/100/100/100 |
| Large reversal backlash | 2/73/32/95 | 0/81/12/63 | 0/87/28/50 |
| Sensor latency | — | 96/95/72/100 | — |

The backlash+gain model included a fixed offset, command gain error,
and a dead zone after direction reversal. Offset jumps changed the
hidden offset once during some episodes. Cyclic error varied with
joint angle. Sensor latency delayed measured positions by 2–4 steps.
These were **simulated** error models; none was identified from this
physical robot.

The large-backlash case is a caution: green DAgger fell to **16/50**
on a separate no-error set, cylinder to **24/50**, while cube remained
50/50. Cylinder and cube also required correction clips of **28.4°**
and **57.9°**, far above a plausible small residual. A cylinder
zero-error-mixed repair improved 20/50 to 43/50 on a new set but still
regressed versus SmolVLA's 49/50. Load-dependent sag and combined-servo
studies were stopped before training by the pre-specified success-band
or ≤6° correction feasibility gates. A nonmonotone direction-lag
pilot trained but scored only 4/50 on selection; a revised monotone lag
study stopped before training because SmolVLA stayed at 49–50/50
across its validation sweep.

**Independent cyclic repeat:** after the exploratory cylinder cyclic
100-episode result, a fresh pre-registered 200-episode test reused
the same locked model and classical settings. SmolVLA / DAgger /
constant-offset Kalman / a sinusoidal model-aware estimator scored
**158 / 196 / 190 / 195**. Direct DAgger-versus-Kalman exact paired
p was **0.146**, and DAgger-versus-sinusoidal p was **1.0**.
The initial DAgger-over-Kalman signal did not meet its replication
criterion.

## Current deployment choice and limits

The real-robot branch deploys the original **constant-offset** 10-step
DAgger estimators, one per task, because their simulator zero-offset
scores were near SmolVLA's baseline and their weights and paired tests
were locked. No artificial error is injected on the robot. We have
no real-robot DAgger success count yet. Sang's earlier real-robot PPO
rollouts established a working camera/control path, but their summary
JSONs contain step timing, not an automatic task-success score.
See [DAGGER_RUNBOOK.md](real_robot/DAGGER_RUNBOOK.md) for the real-robot
commands and log format.
