"""Workspace buffer management

WORKSPACE LIFECYCLE:
1. Create manager: create_workspace_manager(device, buffer_pool)
2. Get/create workspace: workspace_get_or_create(manager, model_params, batch_size, seq_len)
3. Use workspace buffers for forward/backward passes
4. Optionally release: workspace_release(manager, batch_size, seq_len)
5. LRU eviction: workspace_release_lru(manager, keep_count)

MEMORY MANAGEMENT:
- Workspaces are cached by (batch_size, seq_len) for reuse
- Buffers are taken from pool on creation, returned on release
- LRU eviction prevents unbounded memory growth
- All workspaces automatically released on manager cleanup
"""

from dataclasses import fields
from typing import Dict

from gpu_buffer import pool_release_buffer, pool_take_buffer_2d
from gpu_types import (
    BufferPool,
    Device,
    GPUModelParams,
    WorkspaceBuffers,
    WorkspaceManager,
)

# ============================================================================
# WORKSPACE MANAGER
# ============================================================================


def create_workspace_manager(
    device: Device, buffer_pool: BufferPool
) -> WorkspaceManager:
    """Create workspace manager state.

    Args:
        device: GPU device state
        buffer_pool: Buffer pool for workspace allocation

    Returns:
        New workspace manager with empty workspace cache
    """
    return WorkspaceManager(device=device, buffer_pool=buffer_pool)


def workspace_get_or_create(
    manager: WorkspaceManager,
    model_params: GPUModelParams,
    batch_size: int,
    seq_len: int,
) -> WorkspaceBuffers:
    """Get existing workspace or create new one (mutation).

    This function MUTATES manager.active_workspaces if workspace doesn't exist.
    Returns workspace buffers for use in forward/backward passes.

    This is the preferred API for workspace access - combines allocation
    and retrieval into a single call.

    Args:
        manager: Workspace manager state (MUTATED if workspace doesn't exist)
        model_params: Model parameters (for determining buffer sizes)
        batch_size: Batch size for this workspace
        seq_len: Sequence length for this workspace

    Returns:
        Workspace buffers for the specified dimensions

    Raises:
        ValueError: If batch_size or seq_len are invalid
        MemoryError: If buffer pool cannot allocate required memory
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    key = (batch_size, seq_len)

    # If workspace already exists, return it
    if key in manager.active_workspaces:
        return manager.active_workspaces[key]

    # Validate model_params
    if model_params.embedding.shape[0] <= 0 or model_params.embedding.shape[1] <= 0:
        raise ValueError(f"Invalid embedding shape: {model_params.embedding.shape}")

    # Create new workspace by taking buffers from pool
    embedding_dim = model_params.embedding.shape[1]
    vocab_size = model_params.embedding.shape[0]
    total_tokens = batch_size * seq_len
    pool = manager.buffer_pool

    try:
        workspace = WorkspaceBuffers(
            x_buffer_a=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            x_buffer_b=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            x_norm1=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            x_norm2=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            Q=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            K=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            V=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            attn_out_pre=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            attn_out=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            x_with_attn=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            hidden=pool_take_buffer_2d(pool, total_tokens, 4 * embedding_dim),
            hidden_bias=pool_take_buffer_2d(pool, total_tokens, 4 * embedding_dim),
            hidden_gelu=pool_take_buffer_2d(pool, total_tokens, 4 * embedding_dim),
            ffn_out=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            ffn_out_bias=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            logits=pool_take_buffer_2d(pool, total_tokens, vocab_size),
            grad_logits=pool_take_buffer_2d(pool, total_tokens, vocab_size),
            grad_embedding=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            grad_x=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            grad_attn=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            grad_ffn=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            grad_ln1=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
            grad_ln2=pool_take_buffer_2d(pool, total_tokens, embedding_dim),
        )
    except (ValueError, MemoryError) as e:
        # If allocation fails, ensure we don't have a partial workspace
        raise MemoryError(
            f"Failed to allocate workspace for batch_size={batch_size}, seq_len={seq_len}: {e}"
        ) from e

    # Store in manager (mutation)
    manager.active_workspaces[key] = workspace
    return workspace


def workspace_release(manager: WorkspaceManager, batch_size: int, seq_len: int) -> None:
    """Return workspace buffers to pool (mutation).

    This function MUTATES manager.active_workspaces by removing the workspace
    and returning all buffers to the pool. Returns None to signal mutation.

    Args:
        manager: Workspace manager state (MUTATED)
        batch_size: Batch size for workspace to release
        seq_len: Sequence length for workspace to release
    """
    key = (batch_size, seq_len)
    if key in manager.active_workspaces:
        workspace = manager.active_workspaces[key]
        # Release all buffers back to pool by iterating over dataclass fields
        for field in fields(workspace):
            buffer = getattr(workspace, field.name)
            pool_release_buffer(manager.buffer_pool, buffer)
        del manager.active_workspaces[key]


def workspace_clear_all(manager: WorkspaceManager) -> None:
    """Release all workspaces (mutation).

    This function MUTATES manager.active_workspaces by removing all workspaces
    and returning all buffers to the pool. Returns None to signal mutation.

    Args:
        manager: Workspace manager state (MUTATED)
    """
    for key in list(manager.active_workspaces.keys()):
        workspace_release(manager, *key)


def workspace_release_lru(manager: WorkspaceManager, keep_count: int = 2) -> int:
    """Release least recently used workspaces (mutation).

    This implements LRU eviction to prevent unbounded memory growth during training.
    Currently uses workspace size (batch_size * seq_len) as a proxy for LRU.
    In production, would track actual access times.

    This function MUTATES manager.active_workspaces by releasing workspaces.

    Args:
        manager: Workspace manager state (MUTATED)
        keep_count: Number of workspaces to keep (most recently accessed)

    Returns:
        Number of workspaces released

    Raises:
        ValueError: If keep_count is negative
    """
    if keep_count < 0:
        raise ValueError(f"keep_count must be non-negative, got {keep_count}")

    if len(manager.active_workspaces) <= keep_count:
        return 0

    # For simplicity, release smallest workspaces first
    # In production, would track access times for true LRU
    sorted_keys = sorted(
        manager.active_workspaces.keys(),
        key=lambda k: k[0] * k[1],  # Sort by batch_size * seq_len
    )

    num_to_release = len(sorted_keys) - keep_count
    released = 0

    for key in sorted_keys[:num_to_release]:
        workspace_release(manager, *key)
        released += 1

    return released


def workspace_get_memory_usage(manager: WorkspaceManager) -> Dict[str, any]:
    """Get memory usage statistics for workspace manager.

    This function does NOT mutate manager - it only reads from it.

    Args:
        manager: Workspace manager state (not mutated)

    Returns:
        Dictionary with:
        - num_workspaces: Number of active workspaces
        - workspace_configs: List of (batch_size, seq_len) tuples
    """
    return {
        "num_workspaces": len(manager.active_workspaces),
        "workspace_configs": list(manager.active_workspaces.keys()),
    }
