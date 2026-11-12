# cube_flip DAgger round-3 10-step predictor

This package is fixed to the model at `docs/residual/results/dagger_cube_flip/round_3/predictor_10step.pt` (SHA256 `9cb75b09aa511145aa7440f80b335165a42e8d370f6226648a395f90a65e4414`).
Its frozen SmolVLA base is `tsangb34/smolvla_tri_cube_flip_100`, training step **100000**, local checkpoint `checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model`.

## Online control at 30 Hz

Call `reset()` at the start of each robot episode. For each control step, obtain the current SmolVLA **absolute** target from its 50-action queue and the measured six-motor state. Both use the order `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`. The five arm values are degrees; the gripper is 0–100. Call `OnlineCorrector.step(measured_state, smolvla_target, zero_based_step_index)` once, then send its `commanded_target` through the robot's normal joint-limit safety layer. `runtime.py` runs on CPU and needs only PyTorch and NumPy.

```python
from runtime import OnlineCorrector
corrector = OnlineCorrector('/path/to/this/package')
corrector.reset()
for step_index in range(episode_horizon):
    measured_state = read_six_motor_values()
    smolvla_target = next_target_from_current_50_action_chunk()
    result = corrector.step(measured_state, smolvla_target, step_index)
    send_with_robot_joint_limits(result['commanded_target'])
```

The **31 features in exact order** are:

1. 0:6 measured joint positions, normalized (state-mid)/half
2. 6:12 current SmolVLA absolute target, normalized (base-mid)/half
3. 12:18 previous commanded target, normalized (previous-mid)/half; at reset previous=state
4. 18:24 previous commanded target minus measured joint, divided by half
5. 24 current zero-based control-step index modulo 50, divided by 50
6. 25:31 previous clipped normalized residual action (six values); zero at reset

The `mid` and `half` vectors in `manifest.json` come from the simulator action limits used during training. Feature entries 0–23 normalize with those vectors; entry 24 is `(step_index % 50)/50`; entries 25–30 are the previous clipped normalized residual. The first step uses measured state for the previous command and zeros for previous residual. The previous ten 31-feature vectors are stacked oldest to newest; the first vector is repeated on the left until ten are available. `normalization.json` gives the 310-element training mean and standard deviation applied before the MLP. The network is 310→256→256→5 with Tanh hidden layers and outputs the estimated five-arm offset in degrees. The true offset is **never** an online input.

The normalized residual is `clip(-predicted_offset / 4°, -1, 1)` on each arm joint and zero on the gripper. The signed arm correction is four degrees times that residual; add it to the current SmolVLA target. The package helper clips the command to the training joint limits before storing it as previous command. Always enforce the real robot's hardware joint limits when sending targets; limit enforcement may slightly change behavior near the limits. Reset the history whenever the SmolVLA runner starts a new episode; do **not** reset at every 50-action chunk.

`manifest.json` records the 1-step model path and hash as a comparison checkpoint, but only the locked 10-step model is deployed here.
