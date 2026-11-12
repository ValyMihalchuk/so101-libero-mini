# Residual calibration project: colleague overview

*Status: 12 November 2026. Simulator results below; DAgger has not yet been tested on the real robot.*

## What we tried

A frozen SmolVLA controls a simulated SO-101 arm at 30 Hz. It predicts chunks of
50 absolute joint targets. We asked whether a small controller, added to each
target, could restore performance when the five arm joints execute biased
targets. We studied **cube flip**, **green push forward**, and **place the
cylinder right of the cube**. Each task has its own SmolVLA checkpoint and
its own correction model. The gripper is not offset in the main experiments.

The main error was a constant, hidden, episode-specific offset of up to
±2° on each arm joint. The correction model sees only 31 features built from
measured joints, current and previous commands, chunk position, and its
previous correction. It does **not** see the true offset. During simulation
training, the true offset can be used as a reward or supervised label;
during evaluation or real operation it is unavailable.

## Main result: constant offsets

These are paired, frozen **100-episode test** results. The two methods in
each row see the same initial states and offset draws. The zero-offset
column uses a separate 50-episode set.

| Task | SmolVLA with ±2° offset | 10-step DAgger | Zero offset, SmolVLA / DAgger |
| --- | ---: | ---: | ---: |
| Green push | 70/100 | **97/100** | 49/50 / 48/50 |
| Cylinder right of cube | 68/100 | **99/100** | 49/50 / 49/50 |
| Cube flip | 63/100 | **100/100** | 50/50 / 50/50 |

On separate 200-episode confirmation sets, SmolVLA versus DAgger
scored **148 vs 196** for green, **145 vs 196** for cylinder, and
**129 vs 200** for cube.

The correction sign is `command = SmolVLA target - predicted offset`,
clipped to ±4° per arm joint and then to joint limits. The gripper target
is unchanged. These are simulated actuator errors, not a measurement of
the real SO-101's errors.

## How we got there

1. **PPO residual policy.** Trained a 2×256 network for 500,000 steps with
   eight environments, using task progress and success as reward. A no-leak
   cube-flip policy improved 62/100 to 81/100 on its frozen test and
   130/200 to 161/200 on confirmation. Matching green and cylinder PPO
   runs did not improve their first tests: green 73/100 to 69/100,
   cylinder 68/100 to 62/100 at the final checkpoints. A later dense
   cancellation-reward variant scored 69/100 against green's 70/100
   baseline and was stopped.
2. **Oracle and offline learning check.** Giving the controller the true
   offset and subtracting it restored roughly zero-error performance on
   the 50 selection episodes: green 49/50, cylinder 49/50, cube 50/50.
   Ordinary supervised predictors had high held-out accuracy but failed
   when their own corrections changed future observations. Their
   selection successes were green 14–16/50, cylinder 0–1/50, and cube
   11–21/50.
3. **DAgger.** Collected new rollouts while using the current predictor,
   then retrained on all collected data. Three rounds of 50 rollouts,
   including 30% random-correction episodes, produced the 10-step
   models in the table. The 1-step versions were weaker, especially on
   cylinder. The model and test choice were fixed before reading test
   outcomes.

## What the follow-up experiments showed

A tuned integral controller and a constant-offset Kalman filter also
recovered **94–100/100** successes under constant offsets. Therefore the
result is **not** that DAgger is needed for this simple error.

We then tested backlash, gain errors, sudden offset jumps, cyclic
position error, sensor latency, and severe reversal backlash. Results
depended on the task and condition. One promising cylinder cyclic test
favored DAgger over a constant-offset Kalman filter (97 vs 88/100), but a
new, pre-registered 200-episode repeat gave 196 vs 190/200
(exact paired p=0.146); a sinusoidal model-aware estimator reached
195/200. This does not establish a reliable DAgger advantage over
well-chosen classical estimators. Severe-backlash runs sometimes favored
DAgger, but some needed corrections far beyond a plausible ±6° arm
limit or damaged zero-error performance. Several load-sag and combined
servo chains stopped at pre-specified feasibility gates before training.

## Real-robot handoff

The separate `real-robot-dagger` branch is based on Sang's working
real-robot control commit. Its launcher uses the **constant-offset**
round-3 10-step DAgger package for each task, not the later stress-test
models. It verifies the SmolVLA and DAgger weight hashes and logs
baseline and DAgger trials. It adds **no synthetic offset** to the robot.
The first physical question is whether these models preserve normal task
performance and whether any naturally occurring tracking error is
corrected. The simulator counts above cannot answer that.

Start with [DAGGER_RUNBOOK.md](real_robot/DAGGER_RUNBOOK.md) for commands,
camera settings, dry-run behavior, and log locations. The companion
[EXPERIMENT_MAP.md](EXPERIMENT_MAP.md) gives the training recipe and
stress-test numbers. Full logs and per-episode records are in
`docs/residual/NOTES.md` and `docs/residual/results/` on the
`residual-rl` branch; only the locked deployment packages are copied
to this real-robot branch.
