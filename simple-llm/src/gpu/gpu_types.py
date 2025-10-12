"""Core data types - plain dataclasses only"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

# ============================================================================
# WGPU TYPE PROTOCOLS
# ============================================================================

# Type-safe protocols for WGPU objects
# These capture the required interface without importing wgpu at type-check time
# Previously used Any which defeated type checking


@runtime_checkable
class GPUBufferProtocol(Protocol):
    """Structural type for wgpu.GPUBuffer - captures required interface"""

    size: int
    usage: int

    def map_sync(self, mode: int) -> None:
        """Map buffer for CPU access"""
        ...

    def read_mapped(self) -> memoryview:
        """Read mapped buffer contents"""
        ...

    def write_mapped(self, data: Any) -> None:
        """Write to mapped buffer"""
        ...

    def unmap(self) -> None:
        """Unmap buffer after CPU access"""
        ...

    def destroy(self) -> None:
        """Explicitly destroy buffer"""
        ...


@runtime_checkable
class GPUQueueProtocol(Protocol):
    """Structural type for wgpu.GPUQueue"""

    def submit(self, command_buffers: Any) -> None:
        """Submit command buffers for execution"""
        ...

    def write_buffer(
        self, buffer: GPUBufferProtocol, buffer_offset: int, data: Any
    ) -> None:
        """Write data directly to buffer"""
        ...

    def on_submitted_work_done(self, callback: Any) -> None:
        """Register callback for work completion"""
        ...


@runtime_checkable
class GPUAdapterProtocol(Protocol):
    """Structural type for wgpu.GPUAdapter"""

    def request_device_sync(self, **kwargs: Any) -> "GPUDeviceProtocol":
        """Request device synchronously"""
        ...

    def request_adapter_info(self) -> Any:
        """Query adapter capabilities"""
        ...


@runtime_checkable
class GPULimitsProtocol(Protocol):
    """Structural type for wgpu.GPU.Limits"""

    max_compute_workgroup_size_x: int


@runtime_checkable
class GPUCommandEncoderProtocol(Protocol):
    begin_compute_pass: Callable
    copy_buffer_to_buffer: Callable
    finish: Callable


@runtime_checkable
class GPUDeviceProtocol(Protocol):
    """Structural type for wgpu.GPUDevice"""

    queue: GPUQueueProtocol
    adapter: GPUAdapterProtocol
    limits: GPULimitsProtocol
    adapter_info: Dict[str, str]

    def create_buffer(
        self, *, size: int, usage: int, mapped_at_creation: bool = False
    ) -> GPUBufferProtocol:
        """Create GPU buffer"""
        ...

    def create_buffer_with_data(self, *, data: Any, usage: int) -> GPUBufferProtocol:
        """Create buffer initialized with data"""
        ...

    def create_shader_module(self, *, code: str) -> Any:
        """Compile shader module from WGSL source"""
        ...

    def create_compute_pipeline(self, *, layout: Any, compute: Any) -> Any:
        """Create compute pipeline"""
        ...

    def create_bind_group(self, *, layout: Any, entries: Any) -> Any:
        """Create bind group for shader resources"""
        ...

    def create_command_encoder(self) -> Any:
        """Create command encoder"""
        ...


# Type aliases using protocols instead of Any
GPUDevice = GPUDeviceProtocol
GPUBuffer = GPUBufferProtocol
GPUAdapter = GPUAdapterProtocol
GPUCommandEncoder = GPUCommandEncoderProtocol

# Other WGPU types that don't need full protocols (internal use only)
GPUBindGroup = Any  # wgpu.GPUBindGroup
GPUComputePipeline = Any  # wgpu.GPUComputePipeline
GPUShaderModule = Any  # wgpu.GPUShaderModule

# ============================================================================
# DEVICE TYPES
# ============================================================================


@dataclass
class GPUConfig:
    """
    Centralized GPU configuration for kernel parameters and memory limits.

    This dataclass is immutable - do not modify fields after creation.
    All parameters can be tuned for different GPU architectures.
    """

    # ========================================================================
    # KERNEL TILE SIZES
    # ========================================================================

    matmul_tile_size: int = 16
    """
    Tile size for matrix multiplication kernels (16x16 default)

    Optimal values:
    - Small GPUs (integrated): 8
    - Mid-range GPUs: 16
    - High-end GPUs: 32

    Constraints:
    - Must be power of 2
    - Shared memory usage: tile_size * tile_size * 2 * 4 bytes
    - 16x16 = 2KB shared memory per tile
    """

    # ========================================================================
    # FLASHATTENTION PARAMETERS
    # ========================================================================

    flash_attn_bc: int = 32
    """
    FlashAttention block size for K/V columns.

    Controls memory tiling for keys and values. Larger values = more shared memory
    usage but fewer kernel iterations.

    Constraints:
    - bc * head_dim * 4 bytes must fit in shared memory
    - bc=32, head_dim=64: 8KB shared memory
    """

    flash_attn_br: int = 32
    """
    FlashAttention block size for Q rows.

    Controls memory tiling for queries. Larger values = more shared memory
    usage but fewer kernel iterations.

    Constraints:
    - br * head_dim * 4 bytes must fit in shared memory
    - br=32, head_dim=64: 8KB shared memory
    """

    flash_attn_max_head_dim: int = 128
    """
    Maximum head dimension for FlashAttention.

    Hardcoded in WGSL due to static shared memory allocation.
    Larger values require kernel recompilation.

    **WARNING**: Changing this requires regenerating kernels!

    Supported values: 64, 128, 256
    - 64: Conservative, works on all GPUs
    - 128: Requires ~50KB workgroup memory
    - 256: Requires ~200KB workgroup memory (high-end only)
    """

    # ========================================================================
    # WORKGROUP SIZES
    # ========================================================================

    default_workgroup_size: int = 256
    """Default workgroup size for simple kernels (e.g., GELU, residual add)"""

    layernorm_workgroup_size: int = 256
    """Workgroup size for layer normalization (uses shared memory reduction)"""

    attention_workgroup_size: int = 256
    """Workgroup size for attention operations"""

    # ========================================================================
    # MEMORY LIMITS
    # ========================================================================

    buffer_pool_max_mb: int = 512
    """Maximum buffer pool size in megabytes"""

    buffer_pool_max_buffer_mb: int = 512
    """Maximum size of individual pooled buffers in MB"""

    workspace_lru_keep_count: int = 2
    """Number of workspace configurations to keep in LRU cache"""

    staging_buffer_threshold_kb: int = 256
    """
    Threshold for using staging buffers (in KB).

    Buffers larger than this use staging buffer pool.
    Buffers smaller use direct queue.write_buffer().
    """

    staging_buffer_max_entries: int = 8
    """Maximum number of different-sized staging buffers to cache"""

    # ========================================================================
    # COMPUTE LIMITS
    # ========================================================================

    max_workgroups_per_dim: int = 65535
    """
    Maximum workgroups per dimension (WGSL limit).

    This is a WebGPU spec limit and should not be changed.
    Used for validation and automatic tiling.
    """

    max_batch_operations: int = 1000
    """
    Maximum operations per batch submission.

    Prevents unbounded command buffer growth.
    Batches are automatically submitted when this limit is reached.
    """

    # ========================================================================
    # NUMERICAL STABILITY
    # ========================================================================

    layernorm_epsilon: float = 1e-5
    """Epsilon for LayerNorm numerical stability (added to variance before sqrt)"""

    optimizer_epsilon: float = 1e-8
    """Epsilon for AdamW optimizer numerical stability"""


# ============================================================================
# BIND GROUP HELPER TYPES
# ============================================================================


@dataclass
class BindGroupEntry:
    """
    Type-safe bind group entry specification

    This dataclass is immutable - do not modify fields after creation.
    """

    binding: int
    buffer: GPUBuffer  # Now type-safe!
    offset: int
    size: int


# ============================================================================
# GPU BUFFER TYPES
# ============================================================================

# Dimension-specific GPU Buffer Types
# These provide compile-time type safety for buffer operations


@dataclass
class GPUBuffer1D:
    """
    1D GPU buffer - for vectors like biases, layer norm params

    This dataclass is immutable - do not modify fields after creation.
    """

    buffer: GPUBuffer  # Now type-safe!
    shape: Tuple[int]
    size: int


@dataclass
class GPUBuffer2D:
    """
    2D GPU buffer - for matrices like weight matrices

    This dataclass is immutable - do not modify fields after creation.
    """

    buffer: GPUBuffer  # Now type-safe!
    shape: Tuple[int, int]
    size: int


GPUBufferAny = Union[GPUBuffer1D, GPUBuffer2D]

# ============================================================================
# PIPELINE CACHE TYPES
# ============================================================================


@dataclass
class PipelineCache:
    """
    Cache for compiled GPU pipelines
    """

    pipelines: Dict[str, GPUComputePipeline] = field(default_factory=dict)
    bind_groups: Dict[int, GPUBindGroup] = field(default_factory=dict)


# ============================================================================
# BATCH OPERATION TYPES
# ============================================================================


@dataclass
class BatchState:
    """
    State for batched GPU operations
    """

    encoder: Optional[GPUCommandEncoder]
    retained_buffers: List[GPUBuffer] = field(default_factory=list)  # Now type-safe!
    enable_profiling: bool = False
    operation_count: int = 0


# ============================================================================
# MODEL PARAMETER TYPES
# ============================================================================


@dataclass
class GPULayerParams:
    """
    Parameters for a single transformer layer

    This dataclass is immutable - do not modify fields after creation.
    """

    # Attention weights
    attn_wq: GPUBuffer2D  # (dim, dim)
    attn_wk: GPUBuffer2D  # (dim, dim)
    attn_wv: GPUBuffer2D  # (dim, dim)
    attn_wo: GPUBuffer2D  # (dim, dim)

    # Feed-forward weights
    ff_w1: GPUBuffer2D  # (dim, 4*dim)
    ff_b1: GPUBuffer1D  # (4*dim,)
    ff_w2: GPUBuffer2D  # (4*dim, dim)
    ff_b2: GPUBuffer1D  # (dim,)

    # Layer norm parameters
    ln_gamma1: GPUBuffer1D  # (dim,)
    ln_beta1: GPUBuffer1D  # (dim,)
    ln_gamma2: GPUBuffer1D  # (dim,)
    ln_beta2: GPUBuffer1D  # (dim,)


@dataclass
class GPUModelParams:
    """
    Complete model parameters

    This dataclass is immutable - do not modify fields after creation.
    """

    embedding: GPUBuffer2D  # (vocab_size, embedding_dim)
    pos_encoding: GPUBuffer2D  # (context_size, embedding_dim)
    layers: List[GPULayerParams]
    final_ln_gamma: GPUBuffer1D
    final_ln_beta: GPUBuffer1D
    output_projection: GPUBuffer2D


@dataclass
class GPUOptimizerState:
    """
    Optimizer state for AdamW
    - List structure: immutable (same length as model layers)
    """

    m_embedding: GPUBuffer2D  # momentum
    v_embedding: GPUBuffer2D  # variance
    m_layers: List[GPULayerParams]
    v_layers: List[GPULayerParams]
    m_output_projection: GPUBuffer2D
    v_output_projection: GPUBuffer2D
    step: int


GPULayerGradients = GPULayerParams
GPUModelGradients = GPUModelParams


# ============================================================================
# PERFORMANCE MONITORING TYPES
# ============================================================================


@dataclass
class KernelTimeStats:
    """
    Statistics for kernel execution times

    This dataclass is immutable - do not modify fields after creation.
    """

    count: int
    total_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float


@dataclass
class PerfStats:
    """
    Complete performance statistics snapshot

    This dataclass is immutable - do not modify fields after creation.
    """

    total_submissions: int
    kernel_times: Dict[str, KernelTimeStats]


@dataclass
class PerfMonitor:
    """
    Performance monitoring state
    """

    kernel_times: Dict[str, List[float]] = field(default_factory=dict)
    memory_usage: Dict[str, Any] = field(default_factory=dict)
    submission_count: int = 0
