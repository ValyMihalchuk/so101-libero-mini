"""Offline feature parity and full fake-robot rollout; never opens hardware."""
from __future__ import annotations
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from so101_libero.env import SO101TabletopEnv, EnvConfig
from so101_libero.residual.calibration_no_leak_env import NoLeakCalibrationResidualEnv
from scripts.residual.real_robot_residual import MODEL, ResidualController, arguments, run

class FakeRunner:
    def reset(self): self.n = 0
    def act(self, obs, task):
        self.n += 1
        return np.array([0, -60, 35, 85, 0, 25], np.float32)

class FakeIO:
    def __init__(self, bias):
        self.state = np.array([0, -60, 35, 85, 0, 25], np.float32)
        self.bias = bias
        self.sent = []
        self.image = np.zeros((256,256,3), np.uint8)
    def connect(self): pass
    def observe(self):
        return type('Obs', (), {'state':self.state.copy(), 'images':{'front':self.image,'wrist':self.image}})()
    def command(self, cmd):
        self.sent.append(cmd.copy())
        self.state = cmd + self.bias
    def disconnect(self): pass


def main():
    env = SO101TabletopEnv('cube_flip', EnvConfig(render=False))
    bounds = (env.action_space.low.copy(), env.action_space.high.copy())
    env.close()
    model = PPO.load(MODEL, device='cpu')
    rng = np.random.default_rng(17)
    controller = ResidualController(*bounds, model=model)
    builder = object.__new__(NoLeakCalibrationResidualEnv)
    builder.lo, builder.hi = bounds
    builder.mid = (builder.lo + builder.hi)/2
    builder.half = np.maximum((builder.hi-builder.lo)/2, 1e-6)
    max_vector = max_action = 0.
    for step in range(150):
        state = rng.uniform(bounds[0], bounds[1]).astype(np.float32)
        base = rng.uniform(bounds[0], bounds[1]).astype(np.float32)
        prev = None if step == 0 else rng.uniform(bounds[0], bounds[1]).astype(np.float32)
        residual = rng.uniform(-1,1,6).astype(np.float32)
        controller.builder.previous_executed = prev
        controller.builder.previous_residual = residual
        controller.builder.step_number = step
        builder.obs = {'observation.state':state}; builder.base_action = base
        builder.previous_executed = prev; builder.previous_residual = residual; builder.step_number = step
        expected = builder._vector()
        actual, act, _, _ = controller.step(state, base, .5)
        expected_act = model.predict(expected, deterministic=True)[0]
        max_vector = max(max_vector, float(np.max(np.abs(expected-actual))))
        max_action = max(max_action, float(np.max(np.abs(expected_act-act))))
    assert max_vector == 0 and max_action == 0
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ('smolvla','residual'):
            a = arguments(['--mode',mode,'--policy','fake','--duration','15','--fps','30',
                           '--trial-id',mode,'--log-dir',tmp,'--inject-offset','shoulder_pan:+1'])
            io = FakeIO(np.array([.5,0,0,0,0,0],np.float32))
            rows = run(a, io=io, runner=FakeRunner(), model=model, bounds=bounds)
            assert len(rows) == 450 and len(io.sent) == 450
            if mode == 'smolvla':
                assert all(np.array_equal(np.asarray(row[21:27],np.float32)-a.offset,
                                          np.asarray(row[9:15],np.float32)) for row in rows)
            summary = json.loads((Path(tmp)/mode/'summary.json').read_text())
            assert summary['steps'] == 450
            assert (Path(tmp)/mode/'policy_view.mp4').stat().st_size > 0
    print(json.dumps({'parity_max_vector_error':max_vector,'parity_max_action_error':max_action,
                      'mock_steps_each_mode':450, 'mock': 'passed'}))

if __name__ == '__main__': main()
