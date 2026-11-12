# Real SO-101: locked DAgger calibration checks

This runner tests the **same round-3 10-step offset estimators** used in the
frozen simulator evaluations. It makes no simulated offset or noise on the
robot. The comparison is between SmolVLA alone and SmolVLA plus DAgger,
starting each run from the same manually reset object and arm poses.

| Task | SmolVLA checkpoint | Simulator zero-offset: SmolVLA / DAgger | Simulator 2°-offset test: SmolVLA / DAgger |
| --- | --- | --- | --- |
| green_push_forward | `tsangb34/smolvla_tri_green_push_100`, step 80000 (`last`) | 49/50 / 48/50 | 70/100 / 97/100 |
| cylinder_right_of_cube | `tsangb34/smolvla_tri_cylinder_right_100`, top-level step 80000 | 49/50 / 49/50 | 68/100 / 99/100 |
| cube_flip | `tsangb34/smolvla_tri_cube_flip_100`, step 100000 | 50/50 / 50/50 | 63/100 / 100/100 |

These are **simulation** results. No DAgger real-robot success has been measured.
Green push had one extra zero-offset failure in simulation, so check it
carefully in the first real trials. The saved predictor SHA256 values are
verified when the runner loads them. The launcher also verifies the exact
SmolVLA weight file SHA256 before it can move the robot.

## What changed from Sang's working path

Sang's `dc850ac` corrected the real PPO runner's limit handling, pacing,
camera defaults, and return-home behavior. This DAgger runner uses the same
`RealRobotIO`, camera settings, crop/deploy step, 30 Hz pacer, video/CSV
logging, and smooth return-home function. Sang's PPO runner is untouched.
DAgger receives the measured six-joint state and each SmolVLA action, keeps
ten steps of the 31 no-leak features, predicts five arm-joint offsets, and
sends `clip(SmolVLA_target - clip(predicted_offset, -4°, +4°), joint_limits)`.
The gripper gets no correction. Joint order is shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, wrist_roll, gripper. The first five values are degrees;
the gripper is 0–100 units. `--strength 1` matches simulator evaluation;
a smaller strength is an optional cautious hardware setting and is logged.
The corrector's history records the actual clipped, scaled command.

The original Sang trials targeted 30 Hz but their summaries show about
16–19 Hz overall because SmolVLA chunk inference pauses the loop. Check
`mean_hz` and `overruns` in each new trial summary.

## Before the robot

Run the fake-robot check on a machine with the project's Python dependencies:

```bash
python scripts/residual/check_real_robot_dagger_offline.py
```

It checks all three model hashes and six steps per task in each mode, with
no hardware connection. On the cluster this passed with `so101lm`.
Use the same calibrated `PORT`, `ROBOT_ID`, cameras, and crop settings
that worked for Sang. The launcher defaults to front camera 3 and wrist 0.
Set `POLICY` to the exact local pretrained_model directory if its default
path is unavailable. The launcher rejects a different model weight hash.

On the robot machine, first inspect the intended commands while reading
sensors **without writing motor targets**:

```bash
TASK_ID=green_push_forward bash scripts/real_smolvla_dagger.sh --mode dagger --dry-run --duration 2 --trial-id green_dry_01
```

Then run paired trials. Manually reset the arm and objects to the same pose
before each trial, and use unique trial IDs; existing logs are never
overwritten. Run baseline and DAgger for each task:

```bash
TASK_ID=green_push_forward bash scripts/real_smolvla_dagger.sh --mode smolvla --trial-id green_base_01
TASK_ID=green_push_forward bash scripts/real_smolvla_dagger.sh --mode dagger --trial-id green_dagger_01
TASK_ID=cylinder_right_of_cube bash scripts/real_smolvla_dagger.sh --mode smolvla --trial-id cylinder_base_01
TASK_ID=cylinder_right_of_cube bash scripts/real_smolvla_dagger.sh --mode dagger --trial-id cylinder_dagger_01
TASK_ID=cube_flip bash scripts/real_smolvla_dagger.sh --mode smolvla --trial-id cube_base_01
TASK_ID=cube_flip bash scripts/real_smolvla_dagger.sh --mode dagger --trial-id cube_dagger_01
```

Set `DURATION`, `MAX_REL`, `PORT`, `ROBOT_ID`, `FRONT_CAM`,
`WRIST_CAM`, `FRONT_CROP`, and `WRIST_CROP` as needed, using the same
values for each pair. Each trial produces
`docs/residual/real_robot_dagger/<task>/<trial-id>/steps.csv`,
`policy_view.mp4`, and `summary.json`. Watch the robot and score task
success manually from video; there is no real-world simulator success flag.
After a full-duration rollout, the robot returns home over 3 seconds by
default, as in Sang's fixed runner. An interruption or error leaves the arm
in place. `RETURN_HOME=0` disables the move.

Do not interpret dry-run motion as an outcome: in dry-run, the robot does
not execute either policy, so the sensor history cannot reproduce an actual
closed-loop rollout. It verifies loading, features, logging, and command
ranges only.
