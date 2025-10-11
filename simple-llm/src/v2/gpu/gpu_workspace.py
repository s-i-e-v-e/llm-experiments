"""Workspace buffer management"""

from typing import Dict, Tuple

from gpu_buffer import pool_get_buffer, pool_release_buffer
from gpu_types import BufferPool, Device, GPUBuffer, GPUModelParams, WorkspaceManager

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
) -> Tuple[WorkspaceManager, Dict[str, GPUBuffer]]:
    """
    Get or create workspace buffers for given batch/sequence size.
    Returns (manager, workspace_dict)
    """
    key = (batch_size, seq_len)

    if key in manager.active_workspaces:
        return manager, manager.active_workspaces[key]

    # Create new workspace
    embedding_dim = model_params.embedding.shape[1]
    vocab_size = model_params.embedding.shape[0]
    total_tokens = batch_size * seq_len

    workspace = {}
    buffer_specs = [
        # Forward buffers
        ("x_buffer_a", (total_tokens, embedding_dim)),
        ("x_buffer_b", (total_tokens, embedding_dim)),
        ("x_norm1", (total_tokens, embedding_dim)),
        ("x_norm2", (total_tokens, embedding_dim)),
        ("Q", (total_tokens, embedding_dim)),
        ("K", (total_tokens, embedding_dim)),
        ("V", (total_tokens, embedding_dim)),
        ("attn_out_pre", (total_tokens, embedding_dim)),
        ("attn_out", (total_tokens, embedding_dim)),
        ("x_with_attn", (total_tokens, embedding_dim)),
        ("hidden", (total_tokens, 4 * embedding_dim)),
        ("hidden_bias", (total_tokens, 4 * embedding_dim)),
        ("hidden_gelu", (total_tokens, 4 * embedding_dim)),
        ("ffn_out", (total_tokens, embedding_dim)),
        ("ffn_out_bias", (total_tokens, embedding_dim)),
        ("logits", (total_tokens, vocab_size)),
        # Backward buffers
        ("grad_logits", (total_tokens, vocab_size)),
        ("grad_embedding", (total_tokens, embedding_dim)),
        ("grad_x", (total_tokens, embedding_dim)),
        ("grad_attn", (total_tokens, embedding_dim)),
        ("grad_ffn", (total_tokens, embedding_dim)),
        ("grad_ln1", (total_tokens, embedding_dim)),
        ("grad_ln2", (total_tokens, embedding_dim)),
    ]

    # Allocate all buffers from pool
    for name, shape in buffer_specs:
        manager.buffer_pool, buffer = pool_get_buffer(manager.buffer_pool, shape)
        workspace[name] = buffer

    manager.active_workspaces[key] = workspace
    return manager, workspace


def workspace_release(
    manager: WorkspaceManager, batch_size: int, seq_len: int
) -> WorkspaceManager:
    """Return workspace buffers to pool. Returns updated manager"""
    key = (batch_size, seq_len)
    if key in manager.active_workspaces:
        workspace = manager.active_workspaces[key]
        for buffer in workspace.values():
            manager.buffer_pool = pool_release_buffer(manager.buffer_pool, buffer)
        del manager.active_workspaces[key]

    return manager


def workspace_clear_all(manager: WorkspaceManager) -> WorkspaceManager:
    """Release all workspaces. Returns updated manager"""
    for key in list(manager.active_workspaces.keys()):
        manager = workspace_release(manager, *key)
    return manager
