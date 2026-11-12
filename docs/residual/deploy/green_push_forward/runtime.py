"""Standalone, no-simulator online 10-step offset corrector for SO-101 targets."""
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class OffsetMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(310,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,5))
    def forward(self,x):return self.net(x)


class OnlineCorrector:
    """Call once per 30 Hz control step after obtaining that step's SmolVLA target."""
    def __init__(self,package_dir):
        root=Path(package_dir)
        self.manifest=json.loads((root/'manifest.json').read_text())
        model_path=root/'predictor_10step.pt'
        digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest!=self.manifest['predictor_10step_sha256']:
            raise ValueError('deployed predictor SHA256 does not match manifest')
        saved=torch.load(model_path,map_location='cpu',weights_only=False)
        if int(saved['history'])!=10:raise ValueError('package is not a 10-step predictor')
        self.mean=np.asarray(saved['input_mean'],np.float32)
        self.std=np.asarray(saved['input_std'],np.float32)
        if self.mean.shape!=(310,) or self.std.shape!=(310,) or np.any(self.std<=0):
            raise ValueError('invalid predictor normalization')
        self.mid=np.asarray(self.manifest['joint_mid'],np.float32)
        self.half=np.asarray(self.manifest['joint_half_range'],np.float32)
        self.low=np.asarray(self.manifest['joint_target_low'],np.float32)
        self.high=np.asarray(self.manifest['joint_target_high'],np.float32)
        self.model=OffsetMLP()
        self.model.load_state_dict(saved['model_state'])
        self.model.eval()
        self.reset()

    def reset(self):
        self.history=deque(maxlen=10)
        self.previous_command=None
        self.previous_residual=np.zeros(6,np.float32)

    @torch.inference_mode()
    def step(self,measured_joint,smolvla_target,step_index):
        state=np.asarray(measured_joint,np.float32).reshape(6)
        base=np.asarray(smolvla_target,np.float32).reshape(6)
        prev=state if self.previous_command is None else self.previous_command
        feature=np.concatenate(((state-self.mid)/self.half,(base-self.mid)/self.half,
            (prev-self.mid)/self.half,(prev-state)/self.half,
            np.array([(int(step_index)%50)/50],np.float32),self.previous_residual)).astype(np.float32)
        assert feature.shape==(31,)
        self.history.append(feature.copy())
        values=list(self.history)
        stacked=np.concatenate([values[0]]*(10-len(values))+values).astype(np.float32)
        estimate=self.model(torch.from_numpy(((stacked-self.mean)/self.std)[None]))[0].numpy()
        residual=np.zeros(6,np.float32)
        residual[:5]=np.clip(-estimate/4.0,-1.0,1.0)
        correction=np.zeros(6,np.float32);correction[:5]=4.0*residual[:5]
        commanded=base+correction
        if self.manifest['simulator_preclips_command']:
            commanded=np.clip(commanded,self.low,self.high)
        self.previous_command=commanded.copy()
        self.previous_residual=residual.copy()
        return {'estimated_offset_deg':estimate.copy(),
                'residual_normalized':residual.copy(),
                'correction_deg':correction.copy(),
                'commanded_target':commanded.copy()}
