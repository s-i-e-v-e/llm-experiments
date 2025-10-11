"""Core data types - plain dataclasses only"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


@dataclass
class GPUBuffer:
    buffer: object
    shape: Tuple[int, ...]
    size: int
    device: object


@dataclass
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


@dataclass
class GPUModelParams:
    embedding: GPUBuffer
    pos_encoding: GPUBuffer
    layers: list  # List of GPULayerParams


@dataclass
class GPUOptimizerState:
    m_embedding: GPUBuffer
    v_embedding: GPUBuffer
    m_layers: list  # List of GPULayerParams (momentum)
    v_layers: list  # List of GPULayerParams (variance)
    step: int


# ============================================================================
# Performance Monitoring Types
# ============================================================================


@dataclass
class KernelTimeStats:
    """Statistics for a single kernel's execution times"""

    count: int
    total_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float


@dataclass
class PerfStats:
    """Performance statistics snapshot"""

    total_submissions: int
    kernel_times: Dict[str, KernelTimeStats]


@dataclass
class PerfMonitor:
    """Performance monitoring state"""

    kernel_times: Dict[str, List[float]] = field(default_factory=dict)
    memory_usage: Dict[str, Any] = field(default_factory=dict)
    submission_count: int = 0


# ============================================================================
# Workspace Management Types
# ============================================================================


@dataclass
class WorkspaceManager:
    """Workspace buffer manager state"""

    device: object
    buffer_pool: object  # Will be BufferPool
    active_workspaces: Dict[tuple, dict] = field(default_factory=dict)


# ============================================================================
# Batch Operation Types
# ============================================================================


@dataclass
class BatchState:
    """State for batched GPU operations"""

    device: object
    encoder: object
    retained_buffers: List[object] = field(default_factory=list)
    enable_profiling: bool = False
    operation_count: int = 0


# ============================================================================
# Buffer Pool Types
# ============================================================================


@dataclass
class BufferInfo:
    """Information about a pooled buffer"""

    buffer: object


@dataclass
class BufferPool:
    """Memory pool state for reusable GPU buffers"""

    device: object
    max_size: int
    pools: Dict[int, List[BufferInfo]] = field(default_factory=dict)
    in_use: Set[int] = field(default_factory=set)


@dataclass
class StagingPool:
    """Staging buffer pool state for CPU<->GPU transfers"""

    device: object
    staging_buffers: Dict[int, object] = field(default_factory=dict)
    max_size: int = 0
