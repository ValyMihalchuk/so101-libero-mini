# SO-101 residual calibration: all experiments at a glance

 The project uses three tasks: green push (`G`), place cylinder
right of cube (`C`), and cube flip (`F`). In the table, `B` is frozen
SmolVLA alone, `P` a residual PPO policy, `D` a 10-step DAgger predictor,
`I` a tuned integral controller, `K` a tuned constant-offset Kalman
filter, and `O` an oracle given the true simulator error. Every score is
successes out of the stated episode count. Different experiment families
used different frozen starts and seeds; **compare methods within a row**.

___
We kept SmolVLA frozen throughout. It produces joint targets in 50-action chunks; our added controller can adjust each target as the robot moves.

  1. Cube-position shifts. We moved the cube beyond SmolVLA’s demonstration positions to test geometric generalization. We measured the baseline, tried a diagnostic first-chunk nudge toward an expert grasp pose,
     and trained a residual PPO network for 500,000 steps on shifted starts. It made only small gains on the shifted sets.

  2. PPO for joint offsets. We added a hidden, constant offset to each of the five arm joints. We trained a separate 2-layer residual PPO network for each task—cube flip, green push, and cylinder placement—with
     500,000 steps, eight environments, and reward for object progress and success. The no-leak networks used 31 features from measured joints and command history, never the true offset. Cube flip improved; green
     push and cylinder did not. An earlier cube model that could see the offset was a simulator-assisted diagnostic.

  3. PPO variants. We tried a cylinder variant with smaller arm corrections and no gripper correction, then stopped it before test evaluation. We also tried a dense reward for cancelling the true offset and a
     critic allowed to see that offset. The green-push test did not improve, so that variant stopped.

  4. Oracle and ordinary supervised predictors. First, we subtracted the true offset in simulation. This nearly restored zero-error success, showing that cancellation could solve much of the problem. We then
     trained 1-step and 10-step networks on 80 separate SmolVLA rollouts, using the true offset as the training label. Although their held-out predictions were accurate, they failed when used online: their
     corrections changed the observations they would see next.

  5. DAgger for constant offsets. To address that online problem, we collected rollouts while applying the current predictor’s corrections. We added three rounds of 50 training episodes, with random corrections in
     30% of each round, and retrained on all data after every round. We trained both 1-step and 10-step networks, then fixed the round-3 10-step choice before testing. It strongly recovered success under ±2°
     offsets on all three tasks while largely preserving zero-offset performance.

  6. Integral and Kalman comparisons. These are simpler controllers that estimate error from previous commands and measured joints. We tuned their settings on selection episodes; they do not need neural-network
     training. 

  7. Actuator-error stress tests. We simulated backlash with gain error, sudden offset jumps, angle-dependent cyclic error, delayed joint measurements, and large backlash. For each condition and task, we trained a
     new, separate DAgger predictor with the same general three-round procedure, and separately tuned integral and Kalman controllers. Outcomes varied: DAgger sometimes led, but some severe-backlash models needed
     very large corrections or performed badly when the error was absent.



## Experiment descriptions

1. **Object-position shift (F):** Move the cube group beyond SmolVLA's
   ±1 cm demonstration placements. Check scripted-expert feasibility,
   measure SmolVLA degradation, try a privileged first-chunk nudge,
   then train a 500k-step PPO residual on shifted starts. This tests
   geometric generalization, not actuator calibration.
2. **Constant joint-offset calibration (G/C/F):** Add a hidden,
   episode-fixed offset to five arm-joint targets. Sweep 0–8° and work
   at 2°. Train separate 500k-step, eight-environment, 2×256 PPO
   residuals with task-progress plus success reward. The no-leak actor
   receives 31 measured/command-history features, never the offset.
   The first F PPO experiment also had a simulator-assisted version
   whose input revealed the offset; report it separately. A green
   variant added a dense cancellation reward and privileged critic.
3. **Oracle and supervised diagnosis (G/C/F):** Cancel the true offset
   through the same clipped action path to measure the recoverable
   performance. Train 1-step and 10-step supervised 2×256 predictors
   on 80 training-seed rollouts, assess held-out error on 20, then
   apply them online on 50 selection episodes. The offline models
   predicted well but often failed after their corrections changed
   future observations.
4. **Constant-offset DAgger (G/C/F):** Starting from those 80 rollouts,
   collect three rounds of 50 new rollouts using the current predictor;
   30% of each round use random ±3° corrections. Refit 1-step and
   10-step networks to all data after each round (Adam 1e-3,
   20 epochs, batch 1024). Lock the round-3 10-step model before
   frozen 100-test and 200-confirmation evaluations. Its correction
   is `-predicted offset`, clipped to ±4° per arm joint; the gripper
   is unchanged.
5. **Classical offset baselines (G/C/F):** Tune four integral gains and
   eight constant-offset Kalman settings on selection episodes only,
   then compare them with the locked DAgger estimator on identical
   test and zero-error states. Both simple controllers work very well
   under the constant-offset model.
6. **Actuator-model stress tests (G/C/F where listed):** Retrain a
   separate DAgger estimator per task and error model, and retune
   integral/Kalman on that model's selection set. Models include
   backlash plus command gain error, sudden offset jumps, cyclic
   position-dependent error, delayed joint observations, and large
   reversal backlash. These are synthetic faults, not measurements
   of our SO-101. The severe-backlash models sometimes improve test
   success but can need implausibly large corrections or damage
   zero-error performance.
7. **Feasibility and replication checks:** Load-dependent sag and a
   combined servo model stopped before training at pre-specified
   validation/clip gates. A nonmonotone direction-lag pilot failed
   online; a monotone lag follow-up stopped before training because
   SmolVLA barely degraded. A fresh 200-episode cylinder cyclic
   replication tested whether DAgger really beats Kalman and a
   sinusoidal model-aware controller.
8. **Real robot:** Sang's working control script and logs established
   the camera/robot command path. The new `real-robot-dagger` launcher
   packages the **constant-offset** DAgger models for all three tasks
   and injects no artificial error. We have **no scored real-robot
   DAgger trials yet**.

## Results

| Experiment / condition | Task | Set | Paired results | Main reading |
| --- | --- | --- | --- | --- |
| Position shift, original starts | F | 50 | B 50, shifted-start PPO 50 | Original performance retained. |
| Position shift, axis 2 cm | F | 40 | B 19, shifted-start PPO 20 | Small change. |
| Position shift, diagonal 1.5 cm | F | 40 | B 17, shifted-start PPO 20 | Small change. |
| Position shift, feasible axes 3 cm | F | 30 | B 0, shifted-start PPO 0 | Neither succeeded. |
| Position shift, feasible axes 4 cm | F | 30 | B 0, shifted-start PPO 0 | Neither succeeded. |
| Privileged first-chunk nudge, previously stalled | F | 40 | 2°: 0; 5°: 5; 10°: 0 | Diagnostic; uses scripted-expert pose. |
| PPO with offset-revealing input | F | test 100 | B 61, selected P 84, final P 87 | Simulator-assisted, not deployable. |
| PPO with 31 no-leak features | F | test 100; confirm 200 | B 62 → final P 81; B 130 → P 161 | Helped cube flip. |
| PPO with 31 no-leak features | G | test 100; confirm 200 | B 73 → final P 69; B 145 → P 147 | No clear improvement. |
| PPO with 31 no-leak features | C | test 100; confirm 200 | B 68 → final P 62; B 149 → P 132 | Regressed. |
| PPO + dense cancellation reward, privileged critic | G | test 100 | B 70, selected P 70, final P 69 | Stopped; C/F not run. |
| PPO cylinder scale/log-std variant B | C | selection only | B 36/50; P 35/50 at 100k, 30/50 at 200k, 35/50 at 300k | Canceled before test. |
| Oracle and offline predictors, ±2° offset | G | selection 50 | B 33, O 49, 1-step 14, 10-step 16 | Oracle works; offline predictors fail online. |
| Oracle and offline predictors, ±2° offset | C | selection 50 | B 36, O 49, 1-step 1, 10-step 0 | Same failure, strongest on C. |
| Oracle and offline predictors, ±2° offset | F | selection 50 | B 34, O 50, 1-step 21, 10-step 11 | Same failure. |
| Constant-offset DAgger | G | test 100; confirm 200; zero 50 | B/D 70/97; 148/196; 49/48 | Strong offset recovery; one extra zero-error failure. |
| Constant-offset DAgger | C | test 100; confirm 200; zero 50 | B/D 68/99; 145/196; 49/49 | Strong offset recovery and zero-error parity. |
| Constant-offset DAgger | F | test 100; confirm 200; zero 50 | B/D 63/100; 129/200; 50/50 | Strong offset recovery and zero-error parity. |
| Classical constant-offset comparison | G | test 100 | B/D/I/K = 71/98/100/100 | Classical methods match or exceed DAgger. |
| Classical constant-offset comparison | C | test 100 | B/D/I/K = 65/98/94/100 | Kalman best here. |
| Classical constant-offset comparison | F | test 100 | B/D/I/K = 61/100/100/100 | All corrections tie. |
| Backlash + gain | G | test 100 | B/D/I/K = 66/97/95/97 | DAgger and Kalman tie. |
| Backlash + gain | C | test 100 | B/D/I/K = 64/96/77/93 | DAgger highest. |
| Backlash + gain | F | test 100 | B/D/I/K = 60/100/97/96 | DAgger highest. |
| Offset jump | G | test 100 | B/D/I/K = 57/100/96/97 | DAgger highest. |
| Offset jump | C | test 100 | B/D/I/K = 54/99/78/87 | DAgger highest. |
| Offset jump | F | test 100 | B/D/I/K = 59/100/95/100 | DAgger and Kalman tie. |
| Cyclic position error | G | test 100 | B/D/I/K = 89/98/100/98 | Integral highest. |
| Cyclic position error | C | test 100 | B/D/I/K = 84/97/83/88 | Exploratory DAgger lead. |
| Cyclic position error | F | test 100 | B/D/I/K = 81/100/100/100 | All corrections tie. |
| Sensor latency | C | test 100 | B/D/I/K = 96/95/72/100 | Kalman highest. |
| Large reversal backlash | G | test 100; zero 50 | B/D/I/K = 2/73/32/95; zero B/D = 50/16 | Kalman wins; DAgger damages no-error behavior. |
| Large reversal backlash | C | test 100; zero 50 | B/D/I/K = 0/81/12/63; zero B/D = 48/24 | DAgger helps stress, but fails zero error; needs 28.4° clip. |
| Large reversal backlash | F | test 100; zero 50 | B/D/I/K = 0/87/28/50; zero B/D = 50/50 | DAgger helps; needs 57.9° clip. |
| Large-backlash zero-error-mixed repair | C | new test 100; zero 50 | stress B/new D/old D/K = 0/78/77/61; zero = 49/43/20/48 | Better zero-error behavior, still regresses vs B. |
| Independent cylinder cyclic replication | C | new test 200 | B/D/K/sinusoidal = 158/196/190/195 | D vs K exact paired p=0.146; no replicated lead. |
| Load-dependent sag | G/C/F | validation gate | No qualifying 30–70% SmolVLA scale | Stopped before training or test. |
| Combined servo error | G/C/F | correction-size gate | Needed correction exceeded 6° | Stopped before training or test. |
| Nonmonotone direction-lag pilot | C | selection 50 | B 14, best K 24, O 49, round-3 D 4 | Pilot model/labels were discontinuous; no frozen test. |
| Monotone servo-lag follow-up | C | validation 50 per scale | B 49, 50, 50 at scales 0.5, 1, 1.5 | No qualifying scale; stopped before training. |
