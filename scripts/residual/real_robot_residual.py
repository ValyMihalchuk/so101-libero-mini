"""Real SO-101 cube_flip rollout with the frozen no-leak PPO correction."""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from real.run_policy import RealRobotIO, CameraSetting, camera_ref, center_square_resize, Observation, JOINT_NAMES, MOTOR_NAMES

INSTRUCTION = 'Pick up the red cube and flip it onto its side.'
MODEL = ROOT / 'docs/residual/results/calibration/no_leak/ppo/final_model.zip'
CONFIG = ROOT / 'docs/residual/results/calibration/no_leak/ppo/config.json'
SCALE = np.array([4, 4, 4, 4, 4, 2], np.float32)


def parse_offset(spec):
    out = np.zeros(6, np.float32)
    if spec:
        for pair in spec.split(','):
            name, sep, value = pair.partition(':')
            if not sep or name not in JOINT_NAMES:
                raise ValueError(f'bad offset {pair!r}; use joint:+2,joint:-2')
            out[JOINT_NAMES.index(name)] += float(value)
    if not np.all(np.isfinite(out)):
        raise ValueError('offset must be finite')
    return out


class ResidualController:
    """Reuse the training implementation of normalization and feature order."""
    def __init__(self, lo, hi, model=None):
        from so101_libero.residual.calibration_no_leak_env import NoLeakCalibrationResidualEnv
        self.builder = object.__new__(NoLeakCalibrationResidualEnv)
        self.builder.lo = np.asarray(lo, np.float32)
        self.builder.hi = np.asarray(hi, np.float32)
        self.builder.mid = (self.builder.lo + self.builder.hi) / 2
        self.builder.half = np.maximum((self.builder.hi - self.builder.lo) / 2, 1e-6)
        self.builder.previous_executed = None
        self.builder.previous_residual = np.zeros(6, np.float32)
        self.builder.step_number = 0
        self.model = model

    def step(self, state, base, strength):
        self.builder.obs = {'observation.state': np.asarray(state, np.float32)}
        self.builder.base_action = np.asarray(base, np.float32)
        vector = self.builder._vector()
        residual = np.zeros(6, np.float32) if self.model is None else np.asarray(
            self.model.predict(vector, deterministic=True)[0], np.float32).reshape(6)
        if not np.all(np.isfinite(residual)):
            raise RuntimeError('nonfinite residual')
        residual = np.clip(residual, -1, 1)
        raw = self.builder.base_action + np.float32(strength) * SCALE * residual
        command = np.clip(raw, self.builder.lo, self.builder.hi).astype(np.float32)
        if np.any(np.abs(command - base) > SCALE + 1e-5):
            raise RuntimeError('residual exceeds trained maximum')
        self.builder.previous_executed = command.copy()
        self.builder.previous_residual = residual.copy()
        self.builder.step_number += 1
        return vector, residual, command, raw


class ReadOnlyRealIO(RealRobotIO):
    """Read calibrated positions and cameras without configuring motors or sending targets."""
    def connect(self):
        self._require_calibration_file()
        self.robot.bus.connect()
        try:
            for camera in self.robot.cameras.values():
                camera.connect()
        except Exception:
            self.disconnect()
            raise

    def observe(self):
        positions = self.robot.bus.sync_read('Present_Position')
        state = np.array([positions[name] for name in JOINT_NAMES], np.float32)
        images = {name: camera.read_latest() for name, camera in self.robot.cameras.items()}
        return Observation(state, images)

    def command(self, q):
        raise RuntimeError('dry-run cannot send motor commands')

    def disconnect(self):
        for camera in self.robot.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if self.robot.bus.is_connected:
            self.robot.bus.disconnect(disable_torque=False)


def arguments(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('smolvla', 'residual'), default='smolvla')
    p.add_argument('--strength', type=float, default=0.5)
    p.add_argument('--inject-offset', default='')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--trial-id', default='trial')
    p.add_argument('--placement-id', default='unspecified')
    p.add_argument('--policy', required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--port', default='/dev/tty.usbmodem5B610366661')
    p.add_argument('--robot-id', default='my_follower_sim')
    p.add_argument('--front-camera', default='2')
    p.add_argument('--wrist-camera', default='0')
    p.add_argument('--front-width', type=int, default=1280)
    p.add_argument('--front-height', type=int, default=720)
    p.add_argument('--wrist-width', type=int, default=320)
    p.add_argument('--wrist-height', type=int, default=240)
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--duration', type=float, default=60)
    p.add_argument('--max-rel', type=float, default=100)
    p.add_argument('--log-dir', type=Path, default=ROOT/'docs/residual/real_robot')
    p.add_argument('--mock', action='store_true', help='offline fake robot and fake images')
    a = p.parse_args(argv)
    if not 0 <= a.strength <= 1 or a.fps <= 0 or a.duration <= 0:
        p.error('strength must be 0..1; fps and duration must be positive')
    a.offset = parse_offset(a.inject_offset)
    return a


def run(a, io=None, runner=None, model=None, bounds=None):
    if bounds is None:
        from so101_libero.env import SO101TabletopEnv, EnvConfig
        env = SO101TabletopEnv('cube_flip', EnvConfig(render=False))
        bounds = (env.action_space.low.copy(), env.action_space.high.copy())
        env.close()
    lo, hi = bounds
    if model is None and a.mode == 'residual':
        from stable_baselines3 import PPO
        if not MODEL.is_file():
            raise FileNotFoundError(MODEL)
        settings = json.loads(CONFIG.read_text())
        if settings['limit_deg'] != 2 or settings['scale'] != SCALE.tolist():
            raise RuntimeError('residual config does not match trained L=2 scale')
        model = PPO.load(MODEL, device='cpu')
    if runner is None:
        from so101_libero.policy.runner import PolicyRunner
        runner = PolicyRunner(a.policy, device=a.device)
        if runner.cfg.type != 'smolvla' or runner.cfg.n_action_steps != 50 or runner.cfg.chunk_size != 50:
            raise RuntimeError('expected SmolVLA with 50-action chunks')
    if io is None:
        from rollout_tolerant import tolerate_camera_stalls
        tolerate_camera_stalls()
        cams = {'front': CameraSetting(camera_ref(a.front_camera), a.front_width, a.front_height),
                'wrist': CameraSetting(camera_ref(a.wrist_camera), a.wrist_width, a.wrist_height)}
        io_type = ReadOnlyRealIO if a.dry_run else RealRobotIO
        io = io_type(a.port, a.robot_id, cams, int(a.fps), a.max_rel)
    controller = ResidualController(lo, hi, model if a.mode == 'residual' else None)
    trial = a.log_dir / a.trial_id
    if trial.exists():
        raise FileExistsError(f'trial already exists: {trial}')
    trial.mkdir(parents=True)
    import cv2
    writer = cv2.VideoWriter(str(trial/'policy_view.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (512, 256))
    if not writer.isOpened():
        raise RuntimeError('video writer failed')
    fields = ['step', 'time_s', 'inference_ms']
    for group in ('measured', 'smolvla', 'residual', 'final', 'injected_offset'):
        fields += [f'{group}_{name}' for name in JOINT_NAMES]
    rows = []
    connected = False
    reason = 'completed'
    start = time.perf_counter()
    try:
        io.connect(); connected = True
        runner.reset()
        for step in range(round(a.duration*a.fps)):
            tic = time.perf_counter()
            obs = io.observe()
            state = np.asarray(obs.state, np.float32)
            policy_obs = {'observation.state': state,
                          'observation.images.front': obs.images['front'],
                          'observation.images.wrist': obs.images['wrist']}
            t0 = time.perf_counter()
            base = np.asarray(runner.act(policy_obs, INSTRUCTION), np.float32).reshape(6)
            infer_ms = 1000*(time.perf_counter()-t0)
            if not np.all(np.isfinite(base)) or np.any(base < lo) or np.any(base > hi):
                raise RuntimeError('SmolVLA action outside limits or nonfinite')
            _, residual, final, raw = controller.step(state, base, a.strength)
            sent = final + a.offset
            if np.any(raw < lo) or np.any(raw > hi) or np.any(sent < lo) or np.any(sent > hi):
                raise RuntimeError('command outside joint limits')
            if np.any(np.abs(sent-base) > SCALE+1e-5):
                raise RuntimeError('sent command differs from SmolVLA by more than trained maximum')
            if not a.dry_run:
                io.command(sent)
            rows.append([step, round(time.perf_counter()-start, 6), infer_ms,
                         *state, *base, *residual, *sent, *a.offset])
            frame = np.concatenate([center_square_resize(obs.images['front']),
                                    center_square_resize(obs.images['wrist'])], axis=1)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            wait = (step+1)/a.fps - (time.perf_counter()-start)
            if wait > 0: time.sleep(wait)
    except KeyboardInterrupt:
        reason = 'interrupted'
    except Exception as exc:
        reason = f'stopped: {exc}'
        raise
    finally:
        writer.release()
        if connected: io.disconnect()
        with (trial/'steps.csv').open('w', newline='') as f:
            w=csv.writer(f); w.writerow(fields); w.writerows(rows)
        (trial/'summary.json').write_text(json.dumps({'mode':a.mode, 'strength':a.strength,
            'dry_run':a.dry_run, 'trial_id':a.trial_id, 'placement_id':a.placement_id,
            'steps':len(rows), 'reason':reason, 'mean_hz':len(rows)/max(time.perf_counter()-start, 1e-9)}, indent=2)+'\n')
    return rows


if __name__ == '__main__':
    run(arguments())
