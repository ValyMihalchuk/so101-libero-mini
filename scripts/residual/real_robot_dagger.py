"""Real SO-101 rollouts with the locked 10-step DAgger offset estimators.

Robot I/O and pacing follow Sang's working real_robot_residual.py (dc850ac).
No artificial offset is injected.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from real.run_policy import (RealRobotIO, CameraSetting, Pacer, camera_ref,
                             center_square_resize, go_home, JOINT_NAMES)
from scripts.residual.real_robot_residual import ReadOnlyRealIO, BASE_SLACK

INSTRUCTIONS = {
    'green_push_forward': 'Push the green semicircular block forward 5 cm.',
    'cylinder_right_of_cube': 'Put the light gray cylinder to the right of the red cube.',
    'cube_flip': 'Pick up the red cube and flip it onto its side.',
}
ARM_SCALE = 4.0


def load_corrector(task):
    package = ROOT / 'docs/residual/deploy' / task
    spec = importlib.util.spec_from_file_location('dagger_deploy_runtime', package / 'runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    corrector = module.OnlineCorrector(package)  # verifies model SHA256
    m = corrector.manifest
    if (m['task_id'] != task or m['joint_order'] != list(JOINT_NAMES)
            or m['history_steps'] != 10 or m['feature_count_per_step'] != 31
            or m['residual_arm_scale_deg'] != ARM_SCALE or m['gripper_residual_action'] != 0.0
            or m['control_hz'] != 30 or m['smolvla_action_chunk_length'] != 50):
        raise ValueError(f'DAgger package metadata does not match {task} real-robot control')
    return corrector


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=INSTRUCTIONS, required=True)
    p.add_argument('--mode', choices=('smolvla', 'dagger'), default='smolvla')
    p.add_argument('--strength', type=float, default=1.0,
                   help='0..1 fraction of DAgger correction; 1 matches simulation evaluation')
    p.add_argument('--dry-run', action='store_true', help='read sensors without motor writes')
    p.add_argument('--trial-id', default='trial')
    p.add_argument('--placement-id', default='unspecified')
    p.add_argument('--policy', required=True)
    p.add_argument('--source-policy', default='')
    p.add_argument('--device', default='mps')
    p.add_argument('--port', default='/dev/tty.usbmodem5B610366661')
    p.add_argument('--robot-id', default='my_follower_sim')
    p.add_argument('--front-camera', default='3')
    p.add_argument('--wrist-camera', default='0')
    p.add_argument('--front-width', type=int, default=1280)
    p.add_argument('--front-height', type=int, default=720)
    p.add_argument('--wrist-width', type=int, default=320)
    p.add_argument('--wrist-height', type=int, default=240)
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--duration', type=float, default=15)
    p.add_argument('--max-rel', type=float, default=100)
    p.add_argument('--log-dir', type=Path, default=ROOT/'docs/residual/real_robot_dagger')
    p.add_argument('--return-home', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--home-seconds', type=float, default=3.0)
    a = p.parse_args(argv)
    if not 0 <= a.strength <= 1 or a.fps != 30 or a.duration <= 0 or a.home_seconds <= 0:
        p.error('strength must be 0..1; fps must be 30; duration and home-seconds must be positive')
    return a


def run(a, io=None, runner=None, corrector=None, bounds=None):
    if bounds is None:
        from so101_libero.env import SO101TabletopEnv, EnvConfig
        env = SO101TabletopEnv(a.task, EnvConfig(render=False))
        bounds = (env.action_space.low.copy(), env.action_space.high.copy())
        env.close()
    lo, hi = (np.asarray(x, np.float32).reshape(6) for x in bounds)
    if corrector is None and a.mode == 'dagger':
        corrector = load_corrector(a.task)
    if corrector is not None:
        if not np.allclose(corrector.low, lo, atol=1e-3) or not np.allclose(corrector.high, hi, atol=1e-3):
            raise ValueError('DAgger joint limits differ from task environment')
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
        io = (ReadOnlyRealIO if a.dry_run else RealRobotIO)(a.port, a.robot_id, cams, int(a.fps), a.max_rel)
    trial = a.log_dir / a.task / a.trial_id
    if trial.exists():
        raise FileExistsError(f'trial already exists: {trial}')
    trial.mkdir(parents=True)
    import cv2
    writer = cv2.VideoWriter(str(trial/'policy_view.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (512, 256))
    if not writer.isOpened():
        raise RuntimeError('video writer failed')
    fields = ['step', 'time_s', 'inference_ms']
    for group in ('measured', 'smolvla', 'estimated_offset', 'correction', 'sent'):
        fields += [f'{group}_{name}' for name in JOINT_NAMES[:5 if group == 'estimated_offset' else 6]]
    rows = []
    connected = False
    reason = 'completed'
    start = time.perf_counter()
    pacer = None
    returned_home = False
    try:
        io.connect(); connected = True
        runner.reset()
        if corrector is not None:
            corrector.reset()
        start = time.perf_counter()
        pacer = Pacer(1.0 / a.fps)
        for step in range(round(a.duration*a.fps)):
            obs = io.observe()
            state = np.asarray(obs.state, np.float32).reshape(6)
            if not np.all(np.isfinite(state)):
                raise RuntimeError('nonfinite measured joint position')
            policy_obs = {'observation.state': state,
                          'observation.images.front': obs.images['front'],
                          'observation.images.wrist': obs.images['wrist']}
            t0 = time.perf_counter()
            base = np.asarray(runner.act(policy_obs, INSTRUCTIONS[a.task]), np.float32).reshape(6)
            infer_ms = 1000*(time.perf_counter()-t0)
            if not np.all(np.isfinite(base)) or np.any(base < lo-BASE_SLACK) or np.any(base > hi+BASE_SLACK):
                raise RuntimeError(f'SmolVLA action nonfinite or more than {BASE_SLACK} outside limits: '
                                   f'{np.round(base, 2).tolist()}')
            estimate = np.zeros(5, np.float32)
            correction = np.zeros(6, np.float32)
            if a.mode == 'dagger':
                output = corrector.step(state, base, step)
                estimate = np.asarray(output['estimated_offset_deg'], np.float32).reshape(5)
                correction = np.asarray(output['correction_deg'], np.float32).reshape(6)
                if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(correction)):
                    raise RuntimeError('nonfinite DAgger output')
                if np.any(np.abs(correction[:5]) > ARM_SCALE+1e-5) or abs(correction[5]) > 1e-5:
                    raise RuntimeError('DAgger correction exceeds clip or moves gripper')
            sent = np.clip(base + a.strength*correction, lo, hi).astype(np.float32)
            if corrector is not None and a.mode == 'dagger':
                # History must record the target actually sent, including strength and clipping.
                corrector.previous_command = sent.copy()
                actual = sent - np.clip(base, lo, hi)
                corrector.previous_residual = np.r_[actual[:5]/ARM_SCALE, 0].astype(np.float32)
            if np.any(np.abs(sent[:5]-np.clip(base[:5], lo[:5], hi[:5])) > ARM_SCALE+1e-5):
                raise RuntimeError('sent arm command exceeds DAgger correction clip')
            if not a.dry_run:
                io.command(sent)
            rows.append([step, round(time.perf_counter()-start, 6), infer_ms,
                         *state, *base, *estimate, *correction, *sent])
            frame = np.concatenate([center_square_resize(obs.images['front']),
                                    center_square_resize(obs.images['wrist'])], axis=1)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            pacer.wait()
        if a.return_home and not a.dry_run:
            go_home(io, fps=a.fps, seconds=a.home_seconds, max_step_deg=6.0, offsets=np.zeros(6))
            returned_home = True
    except KeyboardInterrupt:
        reason = 'interrupted'
    except Exception as exc:
        reason = f'stopped: {exc}'
        raise
    finally:
        writer.release()
        if connected:
            io.disconnect()
        with (trial/'steps.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(fields); w.writerows(rows)
        (trial/'summary.json').write_text(json.dumps({
            'task': a.task, 'mode': a.mode, 'strength': a.strength,
            'smolvla_repo': corrector.manifest['smolvla_repo'] if corrector else None,
            'smolvla_training_step': corrector.manifest['smolvla_training_step'] if corrector else None,
            'predictor_sha256': corrector.manifest['predictor_10step_sha256'] if corrector else None,
            'policy': str(a.policy), 'source_policy': str(a.source_policy),
            'dry_run': a.dry_run, 'trial_id': a.trial_id, 'placement_id': a.placement_id,
            'steps': len(rows), 'reason': reason,
            'mean_hz': len(rows)/max(time.perf_counter()-start, 1e-9),
            'overruns': pacer.overruns if pacer else 0, 'returned_home': returned_home,
        }, indent=2)+'\n')
    return rows


if __name__ == '__main__':
    run(arguments())
