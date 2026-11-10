# Real robot residual offline check

- Branch/worktree: `real-robot-residual` at `/private/tmp/so101-residual`.
- Model: `docs/residual/results/calibration/no_leak/ppo/final_model.zip`, SHA-256 `85469b20688c532492453d73e11fa49607cd6f3b9540f47eb05a01ac1ea6207f`.
- Chosen calibration level: 2 degrees (`docs/residual/results/calibration/selection.json`); arm residual scale 4 degrees, gripper scale 2 units.
- Parity: 150 generated joint/action records passed through `NoLeakCalibrationResidualEnv._vector()` and the real runner's direct reuse of the same method. Maximum vector difference 0; deterministic PPO output difference 0.
- Mock: fake measured joints followed the sent targets with a fixed 0.5 degree bias. Fake 256x256 camera images, 450 steps in each mode, 30 Hz target. Per-step CSV and MP4 were written. SmolVLA mode's final target equalled its baseline action plus the requested 1 degree software offset at every step.
- No hardware was opened or commanded during these checks. Real dry-run uses a read-only bus and camera connection; that hardware path remains untested until the robot is connected.
- The worktree has no local cube-flip SmolVLA checkpoint. Point `POLICY` to the existing final 100k directory. The Mac has `so101lm` but no `lerobot` conda environment, so set `CONDA_ENV=so101lm`.
- Camera index 2/front and 0/wrist are inherited from the working baseline; the user's anticipated 0/1 mapping still needs checking on the actual camera setup.
