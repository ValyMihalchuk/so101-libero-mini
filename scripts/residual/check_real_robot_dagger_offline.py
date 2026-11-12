"""Offline fake-robot check for every real DAgger task; never connects hardware."""
import json
import tempfile
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.residual.real_robot_dagger import INSTRUCTIONS, arguments, load_corrector, run

BASE = np.array([0, -60, 35, 85, 0, 25], np.float32)

class FakeRunner:
    def __init__(self, instruction):
        self.instruction = instruction
        self.calls = 0
    def reset(self):
        self.calls = 0
    def act(self, observation, instruction):
        assert instruction == self.instruction
        self.calls += 1
        return BASE.copy()

class FakeIO:
    def __init__(self):
        self.state = BASE.copy()
        self.sent = []
        self.image = np.zeros((256, 256, 3), np.uint8)
    def connect(self):
        pass
    def observe(self):
        return type('Obs', (), {'state': self.state.copy(),
                                'images': {'front': self.image, 'wrist': self.image}})()
    def command(self, target):
        self.sent.append(np.asarray(target).copy())
    def disconnect(self):
        pass


def main():
    summary = {}
    with tempfile.TemporaryDirectory() as tmp:
        for task, instruction in INSTRUCTIONS.items():
            corrector = load_corrector(task)
            limits = (corrector.low, corrector.high)
            task_counts = {}
            for mode in ('smolvla', 'dagger'):
                a = arguments(['--task', task, '--mode', mode, '--policy', 'fake',
                               '--dry-run', '--duration', '0.2', '--log-dir', tmp,
                               '--trial-id', mode])
                io = FakeIO()
                rows = run(a, io=io, runner=FakeRunner(instruction),
                           corrector=corrector if mode == 'dagger' else None, bounds=limits)
                assert len(rows) == 6 and not io.sent
                sent = np.asarray([r[-6:] for r in rows], np.float32)
                assert np.all(sent >= limits[0]) and np.all(sent <= limits[1])
                assert np.allclose(sent[:, 5], BASE[5])  # gripper unchanged
                if mode == 'smolvla':
                    assert np.allclose(sent, BASE)  # exact no-noise baseline path
                else:
                    assert np.all(np.abs(sent[:, :5] - BASE[:5]) <= 4.0001)
                logged = json.loads((Path(tmp)/task/mode/'summary.json').read_text())
                assert logged['task'] == task and logged['mode'] == mode
                assert logged['steps'] == 6 and not logged['returned_home']
                assert (Path(tmp)/task/mode/'policy_view.mp4').stat().st_size > 0
                task_counts[mode] = len(rows)
            summary[task] = {'predictor_sha256': corrector.manifest['predictor_10step_sha256'],
                             'steps_per_mode': task_counts}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
