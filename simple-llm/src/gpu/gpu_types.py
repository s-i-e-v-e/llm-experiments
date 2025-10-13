"""Core data types - plain dataclasses only"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import wgpu


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


@dataclass
class PipelineCache:
    """
    Cache for compiled GPU pipelines
    """

    pipelines: Dict[str, wgpu.GPUComputePipeline] = field(default_factory=dict)
    bind_groups: Dict[int, wgpu.GPUBindGroup] = field(default_factory=dict)


@dataclass
class BatchState:
    """
    State for batched GPU operations
    """

    encoder: Optional[wgpu.GPUCommandEncoder]
    retained_buffers: List[wgpu.GPUBuffer] = field(
        default_factory=list
    )  # Now type-safe!
    enable_profiling: bool = False
    operation_count: int = 0


@dataclass
class GPUContext:
    device: wgpu.GPUDevice
    config: GPUConfig
    batch_state: BatchState
    pipeline_cache: PipelineCache


@dataclass
class BindGroupEntry:
    """
    Type-safe bind group entry specification

    This dataclass is immutable - do not modify fields after creation.
    """

    binding: int
    buffer: wgpu.GPUBuffer
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

    buffer: wgpu.GPUBuffer
    shape: Tuple[int]
    size: int


@dataclass
class GPUBuffer2D:
    """
    2D GPU buffer - for matrices like weight matrices

    This dataclass is immutable - do not modify fields after creation.
    """

    buffer: wgpu.GPUBuffer
    shape: Tuple[int, int]
    size: int


# Used by functions that do not care about dimensions
GPUBufferAny = Union[GPUBuffer1D, GPUBuffer2D]


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


# ============================================================================
# KV-CACHE TYPES (for autoregressive generation)
# ============================================================================


@dataclass
class KVCacheLayer:
    """
    KV-cache buffers for a single transformer layer.

    Used during autoregressive generation to avoid recomputing
    K and V for previously generated tokens.

    Shape conventions:
    - k_cache: [batch_size, max_seq_len, n_heads, head_dim]
    - v_cache: [batch_size, max_seq_len, n_heads, head_dim]

    During generation:
    - Position [0:current_len] contains valid cached values
    - Position [current_len:max_seq_len] is unused
    """

    k_cache: GPUBuffer2D  # Flattened: [batch_size * max_seq_len, n_heads * head_dim]
    v_cache: GPUBuffer2D  # Flattened: [batch_size * max_seq_len, n_heads * head_dim]


@dataclass
class KVCache:
    """
    Complete KV-cache for all transformer layers.

    Stores cached key and value tensors to accelerate autoregressive generation.

    Attributes:
        layers: KV-cache buffers for each transformer layer
        batch_size: Number of sequences being generated in parallel
        max_seq_len: Maximum sequence length cache can hold
        current_len: Current number of cached tokens (0 to max_seq_len)
        n_heads: Number of attention heads
        head_dim: Dimension per attention head
    """

    layers: List[KVCacheLayer]
    batch_size: int
    max_seq_len: int
    current_len: int  # Mutable: tracks how many positions are filled
    n_heads: int
    head_dim: int


@dataclass
class KVCacheConfig:
    """
    Configuration for KV-cache allocation.

    Used to specify cache parameters before allocation.

    Attributes:
        batch_size: Number of parallel sequences to cache
        max_seq_len: Maximum sequence length to support
        n_layers: Number of transformer layers
        n_heads: Number of attention heads per layer
        head_dim: Dimension per attention head
    """

    batch_size: int
    max_seq_len: int
    n_layers: int
    n_heads: int
    head_dim: int
