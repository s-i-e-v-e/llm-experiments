"""Core data types - plain dataclasses only"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Union


@dataclass
class Device:
    """GPU device wrapper"""

    wgpu_device: object  # The actual wgpu.Device object
    adapter: object = None  # Optional adapter reference


@dataclass
class PipelineCache:
    """Compute pipeline cache state"""

    device: Device
    pipelines: Dict[Tuple[int, int], object] = field(
        default_factory=dict
    )  # (device_id, shader_hash) -> pipeline


# ============================================================================
# Dimension-specific GPU Buffer Types
# ============================================================================


@dataclass
class GPUBuffer1D:
    """1D GPU buffer - for vectors like biases, layer norm params"""

    buffer: object
    shape: Tuple[int]
    size: int
    device: Device


@dataclass
class GPUBuffer2D:
    """2D GPU buffer - for matrices like weight matrices"""

    buffer: object
    shape: Tuple[int, int]
    size: int
    device: Device


@dataclass
class GPUBuffer3D:
    """3D GPU buffer - for batched sequences (batch, seq, dim)"""

    buffer: object
    shape: Tuple[int, int, int]
    size: int
    device: Device


# Union type for when we need to accept any buffer dimension
GPUBuffer = Union[GPUBuffer1D, GPUBuffer2D, GPUBuffer3D]


# ============================================================================
# Model Parameter Types
# ============================================================================


@dataclass
class GPULayerParams:
    """Parameters for a single transformer layer"""

    attn_wq: GPUBuffer2D  # (dim, dim)
    attn_wk: GPUBuffer2D  # (dim, dim)
    attn_wv: GPUBuffer2D  # (dim, dim)
    attn_wo: GPUBuffer2D  # (dim, dim)
    ff_w1: GPUBuffer2D  # (dim, 4*dim)
    ff_b1: GPUBuffer1D  # (4*dim,)
    ff_w2: GPUBuffer2D  # (4*dim, dim)
    ff_b2: GPUBuffer1D  # (dim,)
    ln_gamma1: GPUBuffer1D  # (dim,)
    ln_beta1: GPUBuffer1D  # (dim,)
    ln_gamma2: GPUBuffer1D  # (dim,)
    ln_beta2: GPUBuffer1D  # (dim,)


@dataclass
class GPUModelParams:
    """Complete model parameters"""

    embedding: GPUBuffer2D  # (vocab_size, embedding_dim)
    pos_encoding: GPUBuffer2D  # (context_size, embedding_dim)
    layers: List[GPULayerParams]


@dataclass
class GPUOptimizerState:
    """Optimizer state for AdamW"""

    m_embedding: GPUBuffer2D
    v_embedding: GPUBuffer2D
    m_layers: List[GPULayerParams]  # momentum
    v_layers: List[GPULayerParams]  # variance
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
class WorkspaceBuffers:
    """Typed workspace buffers for a specific batch/sequence size"""

    # Forward pass buffers - 3D tensors (batch*seq, dim) treated as 2D
    x_buffer_a: GPUBuffer2D
    x_buffer_b: GPUBuffer2D
    x_norm1: GPUBuffer2D
    x_norm2: GPUBuffer2D
    Q: GPUBuffer2D
    K: GPUBuffer2D
    V: GPUBuffer2D
    attn_out_pre: GPUBuffer2D
    attn_out: GPUBuffer2D
    x_with_attn: GPUBuffer2D
    hidden: GPUBuffer2D
    hidden_bias: GPUBuffer2D
    hidden_gelu: GPUBuffer2D
    ffn_out: GPUBuffer2D
    ffn_out_bias: GPUBuffer2D
    logits: GPUBuffer2D

    # Backward pass buffers
    grad_logits: GPUBuffer2D
    grad_embedding: GPUBuffer2D
    grad_x: GPUBuffer2D
    grad_attn: GPUBuffer2D
    grad_ffn: GPUBuffer2D
    grad_ln1: GPUBuffer2D
    grad_ln2: GPUBuffer2D


@dataclass
class WorkspaceManager:
    """Workspace buffer manager state"""

    device: Device
    buffer_pool: "BufferPool"  # forward reference
    active_workspaces: Dict[Tuple[int, int], WorkspaceBuffers] = field(
        default_factory=dict
    )


# ============================================================================
# Batch Operation Types
# ============================================================================


@dataclass
class BatchState:
    """State for batched GPU operations"""

    device: Device
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

    device: Device
    max_size: int
    pools: Dict[int, List[BufferInfo]] = field(default_factory=dict)
    in_use: Set[int] = field(default_factory=set)


@dataclass
class StagingPool:
    """Staging buffer pool state for CPU<->GPU transfers"""

    device: Device
    staging_buffers: Dict[int, object] = field(default_factory=dict)
    max_size: int = 0
