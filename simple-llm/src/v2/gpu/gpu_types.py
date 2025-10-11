"""Core data types - plain dataclasses only"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Union

# ============================================================================
# WGPU TYPE ALIASES
# ============================================================================
# Type aliases for WGPU objects to improve type safety
# These represent opaque wgpu-py objects but provide better documentation
# We use Any because wgpu-py doesn't export stable type stubs

# WGPU device and adapter types
WGPUDevice = Any  # wgpu.GPUDevice
WGPUAdapter = Any  # wgpu.GPUAdapter

# WGPU buffer and command types
WGPUBuffer = Any  # wgpu.GPUBuffer
WGPUCommandEncoder = Any  # wgpu.GPUCommandEncoder
WGPUBindGroup = Any  # wgpu.GPUBindGroup
WGPUComputePipeline = Any  # wgpu.GPUComputePipeline
WGPUShaderModule = Any  # wgpu.GPUShaderModule


# ============================================================================
# DEVICE TYPES
# ============================================================================


@dataclass
class Device:
    """GPU device wrapper"""

    wgpu_device: WGPUDevice  # The actual wgpu.Device object
    adapter: WGPUAdapter = None  # Optional adapter reference


# ============================================================================
# BIND GROUP HELPER TYPES
# ============================================================================


@dataclass
class BindGroupEntry:
    """Type-safe bind group entry specification"""

    binding: int
    buffer: WGPUBuffer
    offset: int
    size: int


# ============================================================================
# GPU BUFFER TYPES
# ============================================================================
# Dimension-specific GPU Buffer Types


@dataclass
class GPUBuffer1D:
    """1D GPU buffer - for vectors like biases, layer norm params"""

    buffer: WGPUBuffer
    shape: Tuple[int]
    size: int
    device: Device


@dataclass
class GPUBuffer2D:
    """2D GPU buffer - for matrices like weight matrices"""

    buffer: WGPUBuffer
    shape: Tuple[int, int]
    size: int
    device: Device


@dataclass
class GPUBuffer3D:
    """3D GPU buffer - for batched sequences (batch, seq, dim)"""

    buffer: WGPUBuffer
    shape: Tuple[int, int, int]
    size: int
    device: Device


# Union type for when we need to accept any buffer dimension
GPUBufferAny = Union[GPUBuffer1D, GPUBuffer2D, GPUBuffer3D]


# ============================================================================
# BUFFER POOL TYPES
# ============================================================================


@dataclass
class BufferInfo:
    """Information about a pooled buffer"""

    buffer: WGPUBuffer


@dataclass
class BufferPool:
    """Memory pool state for reusable GPU buffers"""

    device: Device
    max_size: int
    pools: Dict[int, List[BufferInfo]] = field(default_factory=dict)
    in_use: Set[int] = field(default_factory=set)


@dataclass
class StagingPool:
    """Staging buffer pool state for CPU-GPU transfers"""

    device: Device
    staging_buffers: Dict[int, WGPUBuffer] = field(default_factory=dict)
    max_size: int = 0


# ============================================================================
# PIPELINE CACHE TYPES
# ============================================================================


@dataclass
class PipelineCache:
    """Cache for compiled GPU pipelines"""

    device: Device
    pipelines: Dict[str, WGPUComputePipeline] = field(default_factory=dict)
    bind_groups: Dict[int, WGPUBindGroup] = field(default_factory=dict)


# ============================================================================
# BATCH OPERATION TYPES
# ============================================================================


@dataclass
class BatchState:
    """State for batched GPU operations"""

    device: Device
    encoder: WGPUCommandEncoder
    retained_buffers: List[WGPUBuffer] = field(default_factory=list)
    enable_profiling: bool = False
    operation_count: int = 0


# ============================================================================
# MODEL PARAMETER TYPES
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

    m_embedding: GPUBuffer2D  # momentum
    v_embedding: GPUBuffer2D  # variance
    m_layers: List[GPULayerParams]
    v_layers: List[GPULayerParams]
    step: int


# ============================================================================
# WORKSPACE MANAGEMENT TYPES
# ============================================================================


@dataclass
class WorkspaceBuffers:
    """Typed workspace buffers for a specific batch/sequence size"""

    # Forward pass buffers - 3D tensors [batch*seq, dim] treated as 2D
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
    buffer_pool: BufferPool
    active_workspaces: Dict[Tuple[int, int], WorkspaceBuffers] = field(
        default_factory=dict
    )


# ============================================================================
# PERFORMANCE MONITORING TYPES
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
