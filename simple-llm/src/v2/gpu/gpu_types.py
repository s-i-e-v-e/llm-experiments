"""Core data types - plain dataclasses only"""

import dataclasses
from typing import Tuple


@dataclasses.dataclass
class GPUBuffer:
    buffer: object
    shape: Tuple[int, ...]
    size: int
    device: object


@dataclasses.dataclass
class GPULayerParams:
    attn_wq: GPUBuffer
    attn_wk: GPUBuffer
    attn_wv: GPUBuffer
    attn_wo: GPUBuffer
    ff_w1: GPUBuffer
    ff_b1: GPUBuffer
    ff_w2: GPUBuffer
    ff_b2: GPUBuffer
    ln_gamma1: GPUBuffer
    ln_beta1: GPUBuffer
    ln_gamma2: GPUBuffer
    ln_beta2: GPUBuffer


@dataclasses.dataclass
class GPUModelParams:
    embedding: GPUBuffer
    pos_encoding: GPUBuffer
    layers: list  # List of GPULayerParams


@dataclasses.dataclass
class GPUOptimizerState:
    m_embedding: GPUBuffer
    v_embedding: GPUBuffer
    m_layers: list  # List of GPULayerParams (momentum)
    v_layers: list  # List of GPULayerParams (variance)
    step: int
