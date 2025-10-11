"""Core data types - plain dataclasses only"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
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
class WGPUBufferProtocol(Protocol):
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
class WGPUQueueProtocol(Protocol):
    """Structural type for wgpu.GPUQueue"""

    def submit(self, command_buffers: Any) -> None:
        """Submit command buffers for execution"""
        ...

    def write_buffer(
        self, buffer: WGPUBufferProtocol, buffer_offset: int, data: Any
    ) -> None:
        """Write data directly to buffer"""
        ...

    def on_submitted_work_done(self, callback: Any) -> None:
        """Register callback for work completion"""
        ...


@runtime_checkable
class WGPUDeviceProtocol(Protocol):
    """Structural type for wgpu.GPUDevice"""

    queue: WGPUQueueProtocol

    def create_buffer(
        self, *, size: int, usage: int, mapped_at_creation: bool = False
    ) -> WGPUBufferProtocol:
        """Create GPU buffer"""
        ...

    def create_buffer_with_data(self, *, data: Any, usage: int) -> WGPUBufferProtocol:
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


@runtime_checkable
class WGPUAdapterProtocol(Protocol):
    """Structural type for wgpu.GPUAdapter"""

    def request_device_sync(self, **kwargs: Any) -> WGPUDeviceProtocol:
        """Request device synchronously"""
        ...

    def request_adapter_info(self) -> Any:
        """Query adapter capabilities"""
        ...


# Type aliases using protocols instead of Any
WGPUDevice = WGPUDeviceProtocol
WGPUBuffer = WGPUBufferProtocol
WGPUAdapter = WGPUAdapterProtocol

# Other WGPU types that don't need full protocols (internal use only)
WGPUCommandEncoder = Any  # wgpu.GPUCommandEncoder
WGPUBindGroup = Any  # wgpu.GPUBindGroup
WGPUComputePipeline = Any  # wgpu.GPUComputePipeline
WGPUShaderModule = Any  # wgpu.GPUShaderModule


# ... (all previous content unchanged)

# ============================================================================
# DEVICE TYPES
# ============================================================================


@dataclass
class Device:
    """
    GPU device wrapper

    This dataclass is immutable - do not modify fields after creation.
    """

    wgpu_device: WGPUDevice
    adapter: Optional[WGPUAdapter] = None
    config: Optional["GPUConfig"] = None  # Forward reference to avoid circular import

    def __post_init__(self) -> None:
        """Initialize config if not provided"""
        if self.config is None:
            from gpu_config import create_default_config

            object.__setattr__(self, "config", create_default_config())


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
    buffer: WGPUBuffer  # Now type-safe!
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
    The underlying GPU buffer contents may be mutated by operations.
    """

    buffer: WGPUBuffer  # Now type-safe!
    shape: Tuple[int]
    size: int
    device: Device


@dataclass
class GPUBuffer2D:
    """
    2D GPU buffer - for matrices like weight matrices

    This dataclass is immutable - do not modify fields after creation.
    The underlying GPU buffer contents may be mutated by operations.
    """

    buffer: WGPUBuffer  # Now type-safe!
    shape: Tuple[int, int]
    size: int
    device: Device


@dataclass
class GPUBuffer3D:
    """
    3D GPU buffer - for batched sequences (batch, seq, dim)

    This dataclass is immutable - do not modify fields after creation.
    The underlying GPU buffer contents may be mutated by operations.
    """

    buffer: WGPUBuffer  # Now type-safe!
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
    """
    Information about a pooled buffer

    This dataclass is immutable - do not modify fields after creation.
    """

    buffer: WGPUBuffer  # Now type-safe!


@dataclass
class BufferPool:
    """
    Memory pool state for reusable GPU buffers

    MUTATION SEMANTICS:
    - pools: MUTABLE - buffers are added/removed during pool operations
    - inuse: MUTABLE - tracks which buffers are currently taken
    - totalmemorybytes: MUTABLE - updated as buffers are allocated/freed
    - Other fields: immutable configuration
    """

    device: Device
    maxsize: int  # Max size per individual buffer
    pools: Dict[int, List[BufferInfo]] = field(default_factory=dict)
    inuse: Set[int] = field(default_factory=set)
    totalmemorybytes: int = 0  # Current total memory allocated
    maxtotalmemorybytes: int = 0  # Maximum total memory allowed (0 = unlimited)


@dataclass
class StagingPool:
    """
    Staging buffer pool state for CPU-GPU transfers

    MUTATION SEMANTICS:
    - stagingbuffers: MUTABLE - buffers are added during upload/download operations
    - Other fields: immutable configuration
    """

    device: Device
    stagingbuffers: Dict[int, WGPUBuffer] = field(
        default_factory=dict
    )  # Now type-safe!
    maxsize: int = 0
    maxentries: int = 8  # Limit number of different-sized buffers


# ============================================================================
# PIPELINE CACHE TYPES
# ============================================================================


@dataclass
class PipelineCache:
    """
    Cache for compiled GPU pipelines

    MUTATION SEMANTICS:
    - pipelines: MUTABLE - compiled pipelines are cached on first use
    - bindgroups: MUTABLE - bind groups are cached
    - device: immutable reference
    """

    device: Device
    pipelines: Dict[str, WGPUComputePipeline] = field(default_factory=dict)
    bindgroups: Dict[int, WGPUBindGroup] = field(default_factory=dict)


# ============================================================================
# BATCH OPERATION TYPES
# ============================================================================


@dataclass
class BatchState:
    """
    State for batched GPU operations

    MUTATION SEMANTICS:
    - encoder: MUTABLE - set to None after submit_batch is called
    - retainedbuffers: MUTABLE - accumulates buffers during batch operations, cleared after submit
    - operationcount: MUTABLE - incremented for each operation added
    - Other fields: immutable configuration
    """

    device: Device
    encoder: Optional[WGPUCommandEncoder]
    retainedbuffers: List[WGPUBuffer] = field(default_factory=list)  # Now type-safe!
    enableprofiling: bool = False
    operationcount: int = 0


# ============================================================================
# MODEL PARAMETER TYPES
# ============================================================================


@dataclass
class GPULayerParams:
    """
    Parameters for a single transformer layer

    This dataclass is immutable - do not modify fields after creation.
    The underlying GPU buffer contents may be mutated by training operations.
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
    The underlying GPU buffer contents may be mutated by training operations.
    """

    embedding: GPUBuffer2D  # (vocab_size, embedding_dim)
    pos_encoding: GPUBuffer2D  # (context_size, embedding_dim)
    layers: List[GPULayerParams]


@dataclass
class GPUOptimizerState:
    """
    Optimizer state for AdamW

    MUTATION SEMANTICS:
    - All buffer contents are MUTABLE - updated during optimizer steps
    - step: MUTABLE - incremented after each optimizer step
    - List structure: immutable (same length as model layers)
    """

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
    """
    Typed workspace buffers for a specific batch/sequence size

    This dataclass is immutable - do not modify fields after creation.
    The underlying GPU buffer contents are MUTABLE and reused across operations.

    Buffer naming convention:
    - x_buffer_a, x_buffer_b: ping-pong buffers for layer inputs
    - x_norm1, x_norm2: layer norm outputs
    - Q, K, V: attention query/key/value
    - attn_out_pre, attn_out: attention outputs
    - x_with_attn: residual after attention
    - hidden, hidden_bias, hidden_gelu: FFN intermediate activations
    - ffn_out, ffn_out_bias: FFN outputs
    - logits: final predictions
    - grad_*: gradient buffers for backward pass
    """

    # Forward pass buffers
    x_buffer_a: GPUBuffer2D  # (batch*seq, dim) - ping
    x_buffer_b: GPUBuffer2D  # (batch*seq, dim) - pong
    x_norm1: GPUBuffer2D  # (batch*seq, dim)
    x_norm2: GPUBuffer2D  # (batch*seq, dim)
    Q: GPUBuffer2D  # (batch*seq, dim)
    K: GPUBuffer2D  # (batch*seq, dim)
    V: GPUBuffer2D  # (batch*seq, dim)
    attn_out_pre: GPUBuffer2D  # (batch*seq, dim)
    attn_out: GPUBuffer2D  # (batch*seq, dim)
    x_with_attn: GPUBuffer2D  # (batch*seq, dim)
    hidden: GPUBuffer2D  # (batch*seq, 4*dim)
    hidden_bias: GPUBuffer2D  # (batch*seq, 4*dim)
    hidden_gelu: GPUBuffer2D  # (batch*seq, 4*dim)
    ffn_out: GPUBuffer2D  # (batch*seq, dim)
    ffn_out_bias: GPUBuffer2D  # (batch*seq, dim)
    logits: GPUBuffer2D  # (batch*seq, vocab_size)

    # Backward pass buffers
    grad_logits: GPUBuffer2D  # (batch*seq, vocab_size)
    grad_embedding: GPUBuffer2D  # (batch*seq, dim)
    grad_x: GPUBuffer2D  # (batch*seq, dim)
    grad_attn: GPUBuffer2D  # (batch*seq, dim)
    grad_ffn: GPUBuffer2D  # (batch*seq, dim)
    grad_ln1: GPUBuffer2D  # (batch*seq, dim)
    grad_ln2: GPUBuffer2D  # (batch*seq, dim)


@dataclass
class WorkspaceManager:
    """
    Manager for workspace buffer caching

    MUTATION SEMANTICS:
    - activeworkspaces: MUTABLE - workspaces are added/removed
    - bufferpool: immutable reference (but pool itself is mutable)
    """

    device: Device
    bufferpool: BufferPool
    activeworkspaces: Dict[Tuple[int, int], WorkspaceBuffers] = field(
        default_factory=dict
    )


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

    MUTATION SEMANTICS:
    - kernel_times: MUTABLE - timing data is accumulated during profiling
    - memory_usage: MUTABLE - memory stats are recorded
    - submission_count: MUTABLE - incremented on each submission
    """

    kernel_times: Dict[str, List[float]] = field(default_factory=dict)
    memory_usage: Dict[str, Any] = field(default_factory=dict)
    submission_count: int = 0
