"""Workspace buffer management"""

from dataclasses import fields

from gpu_buffer import pool_get_buffer_2d, pool_release_buffer
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


def workspace_get(
    manager: WorkspaceManager,
    model_params: GPUModelParams,
    batch_size: int,
    seq_len: int,
) -> WorkspaceBuffers:
    """
    Get or create workspace buffers for given batch/sequence size.
    Returns workspace_buffers
    """
    key = (batch_size, seq_len)

    if key in manager.active_workspaces:
        return manager.active_workspaces[key]

    # Create new workspace
    embedding_dim = model_params.embedding.shape[1]
    vocab_size = model_params.embedding.shape[0]
    total_tokens = batch_size * seq_len

    pool = manager.buffer_pool

    workspace = WorkspaceBuffers(
        x_buffer_a=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        x_buffer_b=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        x_norm1=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        x_norm2=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        Q=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        K=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        V=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        attn_out_pre=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        attn_out=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        x_with_attn=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        hidden=pool_get_buffer_2d(pool, total_tokens, 4 * embedding_dim),
        hidden_bias=pool_get_buffer_2d(pool, total_tokens, 4 * embedding_dim),
        hidden_gelu=pool_get_buffer_2d(pool, total_tokens, 4 * embedding_dim),
        ffn_out=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        ffn_out_bias=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        logits=pool_get_buffer_2d(pool, total_tokens, vocab_size),
        grad_logits=pool_get_buffer_2d(pool, total_tokens, vocab_size),
        grad_embedding=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        grad_x=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        grad_attn=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        grad_ffn=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        grad_ln1=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
        grad_ln2=pool_get_buffer_2d(pool, total_tokens, embedding_dim),
    )

    manager.active_workspaces[key] = workspace
    return workspace


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
