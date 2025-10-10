"""Workspace buffer management - refactored from gpu_workspace.py"""

from gpu_buffer import pool_get_buffer, pool_release_buffer

# ============================================================================
# WORKSPACE MANAGER (refactored from WorkspaceManager class)
# ============================================================================


def create_workspace_manager(device: object, buffer_pool_state: dict) -> dict:
    """Create workspace manager state"""
    return {
        "device": device,
        "buffer_pool": buffer_pool_state,
        "active_workspaces": {},  # (batch_size, seq_len) -> workspace dict
    }


def workspace_get(
    manager_state: dict, model_params: GPUModelParams, batch_size: int, seq_len: int
) -> Tuple[dict, Dict[str, GPUBuffer]]:
    """
    Get or create workspace buffers for given batch/sequence size.
    Returns (manager_state, workspace_dict)
    """
    key = (batch_size, seq_len)

    if key in manager_state["active_workspaces"]:
        return manager_state, manager_state["active_workspaces"][key]

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
        manager_state["buffer_pool"], buffer = pool_get_buffer(
            manager_state["buffer_pool"], shape
        )
        workspace[name] = buffer

    manager_state["active_workspaces"][key] = workspace
    return manager_state, workspace


def workspace_release(manager_state: dict, batch_size: int, seq_len: int) -> dict:
    """Return workspace buffers to pool. Returns updated manager_state"""
    key = (batch_size, seq_len)
    if key in manager_state["active_workspaces"]:
        workspace = manager_state["active_workspaces"][key]
        for buffer in workspace.values():
            manager_state["buffer_pool"] = pool_release_buffer(
                manager_state["buffer_pool"], buffer
            )
        del manager_state["active_workspaces"][key]

    return manager_state


def workspace_clear_all(manager_state: dict) -> dict:
    """Release all workspaces. Returns updated manager_state"""
    for key in list(manager_state["active_workspaces"].keys()):
        manager_state = workspace_release(manager_state, *key)
    return manager_state
