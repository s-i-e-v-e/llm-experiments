"""Workspace buffer management"""

from dataclasses import fields

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
    """Create workspace manager state"""
    return WorkspaceManager(device=device, buffer_pool=buffer_pool)


def workspace_ensure_allocated(
    manager: WorkspaceManager,
    model_params: GPUModelParams,
    batch_size: int,
    seq_len: int,
) -> None:
    """
    Ensure workspace exists for given batch/sequence dimensions.

    This function MUTATES manager.active_workspaces by creating a new workspace
    if one doesn't exist. Returns None to signal mutation.

    Must be called before workspace_get_buffers().
    """
    key = (batch_size, seq_len)

    # If workspace already exists, nothing to do
    if key in manager.active_workspaces:
        return

    # Create new workspace by taking buffers from pool
    embedding_dim = model_params.embedding.shape[1]
    vocab_size = model_params.embedding.shape[0]
    total_tokens = batch_size * seq_len
    pool = manager.buffer_pool

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

    # Store in manager (mutation)
    manager.active_workspaces[key] = workspace


def workspace_get_buffers(
    manager: WorkspaceManager,
    batch_size: int,
    seq_len: int,
) -> WorkspaceBuffers:
    """
    Get workspace buffers for given dimensions (read-only access).

    IMPORTANT: workspace_ensure_allocated() must be called first.
    This function does NOT mutate manager - it only reads from it.

    Returns WorkspaceBuffers that are stored in manager, but caller
    should treat this as read-only access to shared state.
    """
    key = (batch_size, seq_len)

    if key not in manager.active_workspaces:
        raise KeyError(
            f"Workspace not allocated for batch_size={batch_size}, seq_len={seq_len}. "
            f"Call workspace_ensure_allocated() first."
        )

    return manager.active_workspaces[key]


def workspace_release(manager: WorkspaceManager, batch_size: int, seq_len: int) -> None:
    """Return workspace buffers to pool"""
    key = (batch_size, seq_len)
    if key in manager.active_workspaces:
        workspace = manager.active_workspaces[key]
        # Release all buffers back to pool by iterating over dataclass fields
        for field in fields(workspace):
            buffer = getattr(workspace, field.name)
            pool_release_buffer(manager.buffer_pool, buffer)
        del manager.active_workspaces[key]


def workspace_clear_all(manager: WorkspaceManager) -> None:
    """Release all workspaces"""
    for key in list(manager.active_workspaces.keys()):
        workspace_release(manager, *key)


def workspace_release_lru(manager: WorkspaceManager, keep_count: int = 2) -> int:
    """
    Release least recently used workspaces, keeping only `keep_count` most recent.

    This implements LRU eviction to prevent unbounded memory growth during training.

    Args:
        manager: Workspace manager state (MUTATED)
        keep_count: Number of workspaces to keep (most recently accessed)

    Returns:
        Number of workspaces released
    """
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


def workspace_get_memory_usage(manager: WorkspaceManager) -> Dict[str, int]:
    """
    Get memory usage statistics for workspace manager.

    Returns:
        Dictionary with:
        - num_workspaces: Number of active workspaces
        - workspace_configs: List of (batch_size, seq_len) tuples
    """
    return {
        "num_workspaces": len(manager.active_workspaces),
        "workspace_configs": list(manager.active_workspaces.keys()),
    }
