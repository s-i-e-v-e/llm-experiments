import gpu


class WorkspaceManager:
    """Manages workspace buffers with automatic pooling and reuse"""

    def __init__(self, device):
        self.device = device
        self.buffer_pool = gpu.BufferPool(device)
        self.active_workspaces = {}  # (batch_size, seq_len) -> WorkspaceBuffers

    def get_workspace(self, model_params, batch_size, seq_len):
        """Get or create workspace buffers for given batch/sequence size"""
        key = (batch_size, seq_len)

        if key in self.active_workspaces:
            return self.active_workspaces[key]

        # Create new workspace
        embedding_dim = model_params.embedding.shape[1]
        vocab_size = model_params.embedding.shape[0]
        total_tokens = batch_size * seq_len

        workspace = {
            # Forward buffers
            "x_buffer_a": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "x_buffer_b": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "x_norm1": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "x_norm2": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "Q": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "K": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "V": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "attn_out_pre": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "attn_out": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "x_with_attn": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "hidden": self.buffer_pool.get_buffer((total_tokens, 4 * embedding_dim)),
            "hidden_bias": self.buffer_pool.get_buffer(
                (total_tokens, 4 * embedding_dim)
            ),
            "hidden_gelu": self.buffer_pool.get_buffer(
                (total_tokens, 4 * embedding_dim)
            ),
            "ffn_out": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "ffn_out_bias": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "logits": self.buffer_pool.get_buffer((total_tokens, vocab_size)),
            # Backward buffers
            "grad_logits": self.buffer_pool.get_buffer((total_tokens, vocab_size)),
            "grad_embedding": self.buffer_pool.get_buffer(
                (total_tokens, embedding_dim)
            ),
            "grad_x": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "grad_attn": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "grad_ffn": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "grad_ln1": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
            "grad_ln2": self.buffer_pool.get_buffer((total_tokens, embedding_dim)),
        }

        self.active_workspaces[key] = workspace
        return workspace

    def release_workspace(self, batch_size, seq_len):
        """Return workspace buffers to pool"""
        key = (batch_size, seq_len)
        if key in self.active_workspaces:
            workspace = self.active_workspaces[key]
            for buffer in workspace.values():
                self.buffer_pool.release_buffer(buffer)
            del self.active_workspaces[key]

    def clear_all(self):
        """Release all workspaces"""
        for key in list(self.active_workspaces.keys()):
            self.release_workspace(*key)
