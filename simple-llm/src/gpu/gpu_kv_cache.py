"""KV-cache management for autoregressive generation"""

from typing import List

import numpy as np

from .gpu_buffer import gpu_buffer_2d_create, gpu_buffer_zerofy
from .gpu_kernels import (
    get_attention_with_kv_cache_kernel,
    get_kv_cache_update_kernel,
)
from .gpu_ops import batch_add, validate_buffer_shape_2d
from .gpu_types import (
    GPUBuffer2D,
    GPUContext,
    KVCache,
    KVCacheConfig,
    KVCacheLayer,
)

# ============================================================================
# KV-CACHE CREATION
# ============================================================================


def kv_cache_create(ctx: GPUContext, config: KVCacheConfig) -> KVCache:
    """
    Create KV-cache buffers for all transformer layers.

    Allocates storage for cached key and value tensors across all layers.
    Buffers are initialized to zero and cache starts empty (current_len=0).

    Args:
        ctx: GPU context
        config: KV-cache configuration specifying dimensions

    Returns:
        KVCache with allocated buffers for all layers

    Raises:
        ValueError: If any dimension in config is <= 0

    Example:
        ```
        cache_config = KVCacheConfig(
            batch_size=1,
            max_seq_len=2048,
            n_layers=12,
            n_heads=8,
            head_dim=64
        )
        cache = kv_cache_create(ctx, cache_config)
        ```
    """
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {config.batch_size}")
    if config.max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {config.max_seq_len}")
    if config.n_layers <= 0:
        raise ValueError(f"n_layers must be > 0, got {config.n_layers}")
    if config.n_heads <= 0:
        raise ValueError(f"n_heads must be > 0, got {config.n_heads}")
    if config.head_dim <= 0:
        raise ValueError(f"head_dim must be > 0, got {config.head_dim}")

    embedding_dim = config.n_heads * config.head_dim
    cache_rows = config.batch_size * config.max_seq_len

    layers: List[KVCacheLayer] = []
    for _ in range(config.n_layers):
        k_cache = gpu_buffer_2d_create(ctx, cache_rows, embedding_dim)
        v_cache = gpu_buffer_2d_create(ctx, cache_rows, embedding_dim)

        # Initialize to zero
        gpu_buffer_zerofy(ctx, k_cache)
        gpu_buffer_zerofy(ctx, v_cache)

        layers.append(KVCacheLayer(k_cache=k_cache, v_cache=v_cache))

    return KVCache(
        layers=layers,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        current_len=0,  # Cache starts empty
        n_heads=config.n_heads,
        head_dim=config.head_dim,
    )


def kv_cache_reset(cache: KVCache) -> None:
    """
    Reset KV-cache to empty state without reallocating buffers.

    Sets current_len to 0, indicating cache is empty.
    Does NOT zero the buffer contents (for performance).

    Args:
        cache: KV-cache to reset

    Note:
        This is a lightweight operation that just resets metadata.
        The actual buffer contents are left unchanged but will be
        overwritten during the next update.
    """
    cache.current_len = 0


# ============================================================================
# KV-CACHE UPDATE
# ============================================================================


def kv_cache_update(
    ctx: GPUContext,
    cache: KVCache,
    layer_idx: int,
    new_k: GPUBuffer2D,
    new_v: GPUBuffer2D,
) -> None:
    """
    Update KV-cache with newly computed K and V for one layer.

    Copies the new key/value tensors into the cache at position cache.current_len.
    This should be called after computing Q, K, V projections during generation.

    Args:
        ctx: GPU context
        cache: KV-cache to update
        layer_idx: Layer index (0 to n_layers-1)
        new_k: Newly computed K [batch_size, 1, embedding_dim]
        new_v: Newly computed V [batch_size, 1, embedding_dim]

    Raises:
        ValueError: If layer_idx out of range or cache is full
        ValueError: If buffer shapes don't match expected dimensions

    Note:
        - Mutates cache by writing to buffers at current_len position
        - Caller must increment cache.current_len after updating all layers
        - new_k and new_v should have shape [batch_size, 1, embedding_dim]
          representing the single new token being generated

    Example:
        ```
        # During generation loop:
        for layer_idx in range(n_layers):
            # Compute Q, K, V for new token
            k_new = matmul(ln_out, layer.wk, ...)
            v_new = matmul(ln_out, layer.wv, ...)

            # Update cache
            kv_cache_update(ctx, cache, layer_idx, k_new, v_new)

        # After all layers updated
        cache.current_len += 1
        ```
    """
    if layer_idx < 0 or layer_idx >= len(cache.layers):
        raise ValueError(f"layer_idx {layer_idx} out of range [0, {len(cache.layers)})")

    if cache.current_len >= cache.max_seq_len:
        raise ValueError(
            f"Cache is full: current_len={cache.current_len}, "
            f"max_seq_len={cache.max_seq_len}"
        )

    embedding_dim = cache.n_heads * cache.head_dim

    # Validate new K/V shapes: [batch_size, 1, embedding_dim]
    expected_rows = cache.batch_size * 1  # Single token
    validate_buffer_shape_2d(new_k, (expected_rows, embedding_dim), "new_k")
    validate_buffer_shape_2d(new_v, (expected_rows, embedding_dim), "new_v")

    layer_cache = cache.layers[layer_idx]

    # FIXME: Kernel needs to be fixed to handle max_seq_len properly
    # Current kernel signature is incomplete
    # For now, we'll prepare parameters but this will fail at runtime

    params = np.array(
        [
            cache.batch_size,
            cache.current_len,  # Position to write to
            embedding_dim,
            cache.max_seq_len,  # FIXME: Kernel needs this in struct
        ],
        dtype=np.uint32,
    )

    # Dispatch kernel to copy new_k and new_v to cache at current_len
    batch_add(
        ctx,
        get_kv_cache_update_kernel(ctx),
        params,
        [new_k, new_v, layer_cache.k_cache, layer_cache.v_cache],
        (embedding_dim + 255) // 256,  # workgroups_x: one per dimension
        cache.batch_size,  # workgroups_y: one per batch element
        1,
    )


# ============================================================================
# ATTENTION WITH KV-CACHE
# ============================================================================


def attention_with_kv_cache(
    ctx: GPUContext,
    cache: KVCache,
    layer_idx: int,
    q_new: GPUBuffer2D,
    output: GPUBuffer2D,
) -> None:
    """
    Compute attention for new query token using cached K/V.

    During autoregressive generation, only the query for the new token
    needs to be computed. Keys and values for all previous tokens are
    retrieved from the cache.

    Args:
        ctx: GPU context
        cache: KV-cache containing cached K/V for all previous tokens
        layer_idx: Layer index (0 to n_layers-1)
        q_new: Query for new token [batch_size, 1, embedding_dim]
        output: Attention output [batch_size, 1, embedding_dim]

    Raises:
        ValueError: If layer_idx out of range or cache is empty
        ValueError: If buffer shapes don't match
        ValueError: If current_len exceeds workgroup capacity

    Note:
        - Assumes causal masking (can only attend to positions <= current_len)
        - Uses numerically stable softmax
        - Limited by workgroup_size: current_len must be <= workgroup_size
          (typically 256 or 512 tokens)
        - For longer sequences, use chunked attention or FlashAttention variant

    Example:
        ```
        # During generation:
        q_new = matmul(ln_out, layer.wq, ...)  # Compute Q for new token
        attn_out = buffer_create(...)          # Output buffer

        attention_with_kvcache(
            ctx, cache, layer_idx, q_new, attn_out
        )
        ```
    """
    if layer_idx < 0 or layer_idx >= len(cache.layers):
        raise ValueError(f"layer_idx {layer_idx} out of range [0, {len(cache.layers)})")

    if cache.current_len == 0:
        raise ValueError("Cache is empty: current_len=0")

    # Check workgroup limit
    workgroup_size = ctx.config.attention_workgroup_size
    if cache.current_len > workgroup_size:
        raise ValueError(
            f"current_len ({cache.current_len}) exceeds maximum supported "
            f"sequence length ({workgroup_size}) for attention_with_kv_cache. "
            f"Use chunked attention or increase workgroup_size in config."
        )

    embedding_dim = cache.n_heads * cache.head_dim

    # Validate shapes
    expected_q_rows = cache.batch_size * 1  # Single token
    validate_buffer_shape_2d(q_new, (expected_q_rows, embedding_dim), "q_new")
    validate_buffer_shape_2d(output, (expected_q_rows, embedding_dim), "output")

    layer_cache = cache.layers[layer_idx]

    params = np.array(
        [
            cache.batch_size,
            cache.current_len,  # Number of valid cached positions
            cache.n_heads,
            cache.head_dim,
        ],
        dtype=np.uint32,
    )

    # Dispatch attention kernel
    # One workgroup per (batch, head) pair
    batch_add(
        ctx,
        get_attention_with_kv_cache_kernel(ctx),
        params,
        [q_new, layer_cache.k_cache, layer_cache.v_cache, output],
        cache.n_heads,  # workgroups_x: one per head
        cache.batch_size,  # workgroups_y: one per batch
        1,
    )
