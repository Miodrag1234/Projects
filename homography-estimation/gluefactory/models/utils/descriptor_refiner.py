"""
Learnable Descriptor Refiner.

A small residual MLP that sits between the feature extractor and the
matcher. Initialized to the identity (last linear is zero), so loading
official pre-trained matcher weights gives the exact baseline behavior
at iteration 0. Training then learns small corrections on top of the
hand-crafted descriptors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


class DescriptorRefiner(BaseModel):
    default_conf = {
        "descriptor_dim": 128,
        "hidden_dim_mult": 2,  # hidden = descriptor_dim * mult
        "normalize_input": True,  # L2 normalize descriptors before refining
        "normalize_output": True,  # L2 normalize after residual
        "dropout": 0.0,
    }

    required_data_keys = ["descriptors0", "descriptors1"]

    def _init(self, conf):
        d = conf.descriptor_dim
        h = d * conf.hidden_dim_mult

        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, h)
        self.act = nn.GELU()
        self.drop = nn.Dropout(conf.dropout) if conf.dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(h, d)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _refine(self, desc: torch.Tensor) -> torch.Tensor:
        if self.conf.normalize_input:
            desc = F.normalize(desc, dim=-1)
        residual = self.fc2(self.drop(self.act(self.fc1(self.norm(desc)))))
        out = desc + residual
        if self.conf.normalize_output:
            out = F.normalize(out, dim=-1)
        return out

    def _forward(self, data: dict) -> dict:
        return {
            "descriptors0": self._refine(data["descriptors0"]),
            "descriptors1": self._refine(data["descriptors1"]),
        }

    def loss(self, pred, data):
        raise NotImplementedError
