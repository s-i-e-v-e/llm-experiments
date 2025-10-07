import json

import numpy as np

# ==============================================================================
# MATH PRIMITIVES & HELPERS
# ==============================================================================


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x, g, b, eps=1e-5):
    # x shape: [batch_size, seq_len, d_model]
    # g, b shape: [d_model]
    mean = np.mean(x, axis=-1, keepdims=True)  # [batch_size, seq_len, 1]
    variance = np.var(x, axis=-1, keepdims=True)  # [batch_size, seq_len, 1]
    return g * (x - mean) / np.sqrt(variance + eps) + b


def linear(x, w, b):
    # x shape: [batch_size, seq_len, input_dim]
    # w shape: [output_dim, input_dim]
    # b shape: [output_dim]
    # Output: [batch_size, seq_len, output_dim]
    return x @ w.T + b


def attention_mask(seq_len):
    """Create causal mask for decoder"""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask * -1e9


# ==============================================================================
# TRANSFORMER COMPONENTS
# ==============================================================================


class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Weight matrices: [H, Dm, Dk] (Transposed for einsum)
        self.w_q = np.random.randn(n_heads, d_model, self.d_k) * 0.02
        self.w_k = np.random.randn(n_heads, d_model, self.d_k) * 0.02
        self.w_v = np.random.randn(n_heads, d_model, self.d_k) * 0.02
        self.w_o = np.random.randn(d_model, d_model) * 0.02
        self.grads = {}

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # FIX #1: Vectorized Linear Projections (Memory Bomb Fix)
        # x is [B, S, Dm]. w_q is [H, Dm, Dk]. Output q/k/v are [H, B, S, Dk]
        q = np.einsum("bjd,hdk->hbjk", x, self.w_q)
        k = np.einsum("bjd,hdk->hbjk", x, self.w_k)
        v = np.einsum("bjd,hdk->hbjk", x, self.w_v)

        # Attention scores (q @ k^T) / sqrt(d_k). Result is [H, B, S, S]
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        if mask is not None:
            scores += mask

        # Softmax and attention output
        attn_weights = softmax(scores, axis=-1)
        attn_output = attn_weights @ v  # [H, B, S, Dk]

        # Combine heads
        # Transpose to [B, S, H, Dk], then reshape to [B, S, Dm]
        attn_output_combined = attn_output.transpose(1, 2, 0, 3).reshape(
            batch_size, seq_len, d_model
        )

        # Output projection
        output = attn_output_combined @ self.w_o.T

        # Cache for backward pass
        self.cache = {
            "x": x,
            "q": q,
            "k": k,
            "v": v,
            "attn_weights": attn_weights,
            "attn_output_combined": attn_output_combined,
        }
        return output

    def backward(self, d_output):
        x, q, k, v, attn_weights, attn_output_combined = (
            self.cache["x"],
            self.cache["q"],
            self.cache["k"],
            self.cache["v"],
            self.cache["attn_weights"],
            self.cache["attn_output_combined"],
        )
        batch_size, seq_len, d_model = x.shape

        # Gradient through output projection
        d_attn_output_combined = d_output @ self.w_o  # [B, S, Dm]
        self.grads["w_o"] = np.einsum(
            "bsd,bsm->md", d_output, attn_output_combined
        )  # [Dm, Dm]

        # CRITICAL FIX: Reshape back to multi-head [H, B, S, Dk]
        d_attn_output = d_attn_output_combined.reshape(
            batch_size, seq_len, self.n_heads, self.d_k
        ).transpose(2, 0, 1, 3)  # [H, B, S, Dk]

        # --- VECTORIZED ATTENTION GRADIENTS ---

        # d_v gradient: d_v = W^T @ dO. [H, B, S, S] @ [H, B, S, Dk] -> [H, B, S, Dk]
        d_v = attn_weights.transpose(0, 1, 3, 2) @ d_attn_output

        # d_attn_weights: dW = dO @ V^T. [H, B, S, Dk] @ [H, B, Dk, S] -> [H, B, S, S]
        d_attn_weights = d_attn_output @ v.transpose(0, 1, 3, 2)

        # Backward through softmax (Must be calculated for the full H, B, S, S)
        d_scores = attn_weights * (
            d_attn_weights
            - np.einsum("hbij,hbij->hbi", attn_weights, d_attn_weights)[
                :, :, :, np.newaxis
            ]
        )
        d_scores = d_scores / np.sqrt(self.d_k)

        # d_q gradient: d_q = dS @ K. [H, B, S, S] @ [H, B, S, Dk] -> [H, B, S, Dk]
        d_q = d_scores @ k

        # d_k gradient: d_k = dS^T @ Q. [H, B, S, S]^T @ [H, B, S, Dk] -> [H, B, S, Dk]
        d_k = d_scores.transpose(0, 1, 3, 2) @ q

        # --- VECTORIZED WEIGHT GRADIENTS ---
        # dW_Q: dQ @ x^T. [H, B, S, Dk] @ [B, Dm, S] -> [H, Dm, Dk]
        self.grads["w_q"] = np.einsum("hbjk,bjl->hkl", d_q, x)
        self.grads["w_k"] = np.einsum("hbjk,bjl->hkl", d_k, x)
        self.grads["w_v"] = np.einsum("hbjk,bjl->hkl", d_v, x)

        # Gradient to input (d_x Accumulation)
        # d_x = sum(dQ @ W_Q) over H. [H, B, S, Dk] @ [H, Dk, Dm] -> [B, S, Dm]
        d_x = np.einsum("hbjk,hkl->bjl", d_q, self.w_q.transpose(0, 2, 1))
        d_x += np.einsum("hbjk,hkl->bjl", d_k, self.w_k.transpose(0, 2, 1))
        d_x += np.einsum("hbjk,hkl->bjl", d_v, self.w_v.transpose(0, 2, 1))

        return d_x


class FeedForward:
    def __init__(self, d_model, d_ff=None):
        d_ff = d_ff or 4 * d_model
        self.w1 = np.random.randn(d_ff, d_model) * 0.02
        self.w2 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
        self.grads = {}

    def forward(self, x):
        self.cache_x = x
        h = linear(x, self.w1, self.b1)
        h_gelu = gelu(h)
        output = linear(h_gelu, self.w2, self.b2)

        self.cache = {"h": h, "h_gelu": h_gelu}
        return output

    def backward(self, d_output):
        x, h, h_gelu = self.cache_x, self.cache["h"], self.cache["h_gelu"]
        batch_size, seq_len, d_model = x.shape

        # Gradient through w2, b2
        # d_output shape: [batch_size, seq_len, d_model]
        # h_gelu shape: [batch_size, seq_len, d_ff]
        # We need: [d_model, d_ff]
        self.grads["w2"] = np.einsum("bij,bik->jk", d_output, h_gelu)  # [d_model, d_ff]
        self.grads["b2"] = d_output.sum(axis=(0, 1))  # [d_model]

        # Gradient through gelu and w1, b1
        d_h_gelu = d_output @ self.w2  # [batch_size, seq_len, d_ff]

        # GELU derivative
        d_gelu = d_h_gelu * (
            0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h**3)))
            + 0.5
            * h
            * (1 - np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h**3)) ** 2)
            * (np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * h**2))
        )

        self.grads["w1"] = np.einsum("bij,bik->jk", d_gelu, x)  # [d_ff, d_model]
        self.grads["b1"] = d_gelu.sum(axis=(0, 1))  # [d_ff]

        # Gradient to input
        d_x = d_gelu @ self.w1  # [batch_size, seq_len, d_model]
        return d_x


class TransformerBlock:
    def __init__(self, d_model, n_heads):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model)

        # Layer norms
        self.ln1_g = np.ones(d_model)
        self.ln1_b = np.zeros(d_model)
        self.ln2_g = np.ones(d_model)
        self.ln2_b = np.zeros(d_model)

        self.grads = {}

    def forward(self, x, mask=None):
        # Self-attention with residual and layer norm
        attn_out = self.attention.forward(x, mask)
        x = layer_norm(x + attn_out, self.ln1_g, self.ln1_b)

        # Feed-forward with residual and layer norm
        ffn_out = self.ffn.forward(x)
        output = layer_norm(x + ffn_out, self.ln2_g, self.ln2_b)

        self.cache_x = x
        return output

    def backward(self, d_output):
        # Backward through second layer norm + residual
        d_ffn = d_output  # Simplified - in practice need layer norm backward
        d_x1 = d_output  # Simplified

        # Backward through FFN
        d_ffn_input = self.ffn.backward(d_ffn)

        # Backward through first layer norm + residual
        d_attn = d_x1  # Simplified
        d_initial = self.attention.backward(d_attn)

        # Combined gradient
        d_input = d_ffn_input + d_initial
        return d_input


# ==============================================================================
# MAIN TRANSFORMER MODEL
# ==============================================================================
import gc


class NumpyTransformer:
    def __init__(self, vocab_size, d_model=128, n_heads=8, n_layers=6, max_seq_len=256):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Better initialization
        scale = 0.02  # Much smaller scale

        # Embeddings
        self.token_embedding = np.random.randn(d_model, vocab_size) * scale
        self.pos_embedding = np.random.randn(max_seq_len, d_model) * scale

        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]

        # Final layer norm and output projection
        self.ln_f_g = np.ones(d_model)
        self.ln_f_b = np.zeros(d_model)
        self.output_proj = np.random.randn(vocab_size, d_model) * scale
        self.output_bias = np.zeros(vocab_size)

        self.grads = {}

    def forward(self, tokens):
        """Forward pass with batching support
        tokens shape: [batch_size, seq_len]
        """
        batch_size, seq_len = tokens.shape

        # FAST: Single NumPy operation
        # token_embedding is [d_model, vocab_size]
        # tokens is [batch_size, seq_len]
        token_embeds = self.token_embedding.T[tokens]
        # Result: [batch_size, seq_len, d_model]

        # Positional embeddings - broadcast to batch
        pos_embeds = self.pos_embedding[:seq_len]  # [seq_len, d_model]
        pos_embeds = pos_embeds[np.newaxis, :, :]  # [1, seq_len, d_model]
        pos_embeds = np.tile(
            pos_embeds, (batch_size, 1, 1)
        )  # [batch_size, seq_len, d_model]

        # Combine embeddings
        x = token_embeds + pos_embeds

        # Create causal mask - same for all batches
        mask = attention_mask(seq_len)

        # Transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # Final layer norm
        x = layer_norm(x, self.ln_f_g, self.ln_f_b)

        # Output projection
        logits = linear(x, self.output_proj, self.output_bias)

        self.cache = {"tokens": tokens, "x_final": x}
        gc.collect()
        return logits  # [batch_size, seq_len, vocab_size]

    def backward(self, d_logits):
        tokens, x_final = self.cache["tokens"], self.cache["x_final"]
        batch_size, seq_len = tokens.shape
        d_model = self.d_model  # Get d_model from the instance

        # Gradient through output projection
        self.grads["output_proj"] = np.einsum(
            "bij,bik->jk", d_logits, x_final
        )  # [vocab_size, d_model]
        self.grads["output_bias"] = d_logits.sum(axis=(0, 1))  # [vocab_size]

        # Gradient through final layer norm (simplified)
        d_x = d_logits @ self.output_proj  # [batch_size, seq_len, d_model]

        # Backward through transformer blocks
        for block in reversed(self.blocks):
            d_x = block.backward(d_x)

        # ----------------------------------------------------------------------
        # FIXED: Gradient through embeddings (Token Embedding Accumulation)
        # Fixes the (256, 5000) vs (5000, 256) ValueError
        # ----------------------------------------------------------------------

        # 1. Initialize a temporary gradient accumulation array: [Vocab, Dm]
        # This is the natural shape for np.add.at accumulation (one row per token ID)
        token_grad_accum = np.zeros((self.vocab_size, d_model), dtype=d_x.dtype)

        # Reshape d_x to [batch_size * seq_len, d_model]
        d_x_flat = d_x.reshape(-1, d_model)

        # Reshape tokens to [batch_size * seq_len]
        tokens_flat = tokens.flatten()

        # 2. Accumulate the gradients using np.add.at
        # np.add.at operates on the rows of the first argument.
        np.add.at(
            token_grad_accum,  # Accumulate into the [Vocab, Dm] array
            tokens_flat,  # Use the token ID as the row index
            d_x_flat,  # Add the corresponding d_x vector
        )

        # 3. Final assignment with the correct transpose
        # self.token_embedding is [Dm, Vocab]. The accumulated gradient is [Vocab, Dm].
        # Transpose it ONCE to match the weight shape before assignment.
        self.grads["token_embedding"] = token_grad_accum.T  # Final shape: [Dm, Vocab]

        gc.collect()
        return d_x

    def update_weights(self, learning_rate):
        """Update all weights using accumulated gradients with gradient clipping"""
        max_grad_norm = 1.0  # Clip gradients to this norm

        # --- Embedding Update (Your previous fix for transpose error is here) ---
        if "token_embedding" in self.grads:
            grad = self.grads["token_embedding"]
            grad_norm = np.linalg.norm(grad)
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
            self.token_embedding -= learning_rate * grad  # Grad is now [Dm, Vocab]

        # --- Output Layer Updates ---
        # ... (Output layer code is fine) ...
        if "output_proj" in self.grads:
            grad = self.grads["output_proj"]
            grad_norm = np.linalg.norm(grad)
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
            self.output_proj -= learning_rate * grad

        if "output_bias" in self.grads:
            grad = self.grads["output_bias"]
            grad_norm = np.linalg.norm(grad)
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
            self.output_bias -= learning_rate * grad

        # --- Transformer Block Updates ---
        for block in self.blocks:
            # --- Attention Weights (W_Q, W_K, W_V, W_O) ---
            if hasattr(block.attention, "grads"):
                for param_name, grad in block.attention.grads.items():
                    # ISSUE 1 & 2 FIX: The keys in grads are 'w_q', 'w_k', 'w_v', 'w_o'
                    param = getattr(
                        block.attention, param_name
                    )  # Simply get the parameter by its name

                    # Only the W_Q, W_K, W_V gradients need a transpose
                    if param_name in ["w_q", "w_k", "w_v"]:  # Corrected check
                        # Parameter shape: [H, Dm, Dk]
                        # Gradient shape: [H, Dk, Dm]
                        grad = grad.transpose(0, 2, 1)  # [H, Dk, Dm] -> [H, Dm, Dk]

                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > max_grad_norm:
                        grad = grad * (max_grad_norm / grad_norm)

                    # Update (Shape mismatch should now be resolved)
                    param -= learning_rate * grad  # This line now works

            # --- FFN Weights (W1, W2, B1, B2) ---
            if hasattr(block.ffn, "grads"):
                # ... (FFN code is fine) ...
                for param_name, grad in block.ffn.grads.items():
                    param = getattr(block.ffn, param_name)

                    # FFN weights are calculated correctly in backward as:
                    # W1: [Dff, Dm], W2: [Dm, Dff], B1: [Dff], B2: [Dm]
                    # No transpose needed here.

                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > max_grad_norm:
                        grad = grad * (max_grad_norm / grad_norm)
                    param -= learning_rate * grad

    def calculate_loss(self, logits, targets):
        """Calculate cross-entropy loss"""
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Stable softmax cross-entropy
        log_probs = logits_flat - np.max(logits_flat, axis=1, keepdims=True)
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=1, keepdims=True))

        loss = -log_probs[np.arange(len(targets_flat)), targets_flat].mean()
        return loss

    def get_loss_gradient(self, logits, targets):
        """Get gradient of loss w.r.t. logits"""
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Softmax probabilities
        probs = softmax(logits_flat, axis=1)

        # Gradient: (probs - one_hot_targets) / batch_size
        d_logits = probs.copy()
        d_logits[np.arange(len(targets_flat)), targets_flat] -= 1
        d_logits = d_logits.reshape(batch_size, seq_len, vocab_size) / batch_size

        return d_logits

    def save_model(self, filepath):
        """Save model to JSON file"""
        model_data = {
            "config": {
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "max_seq_len": self.max_seq_len,
            },
            "weights": {
                "token_embedding": self.token_embedding.tolist(),
                "pos_embedding": self.pos_embedding.tolist(),
                "output_proj": self.output_proj.tolist(),
                "output_bias": self.output_bias.tolist(),
                "ln_f_g": self.ln_f_g.tolist(),
                "ln_f_b": self.ln_f_b.tolist(),
            },
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """Load model from JSON file"""
        with open(filepath, "r") as f:
            model_data = json.load(f)

        config = model_data["config"]
        model = cls(**config)

        weights = model_data["weights"]
        model.token_embedding = np.array(weights["token_embedding"])
        model.pos_embedding = np.array(weights["pos_embedding"])
        model.output_proj = np.array(weights["output_proj"])
        model.output_bias = np.array(weights["output_bias"])
        model.ln_f_g = np.array(weights["ln_f_g"])
        model.ln_f_b = np.array(weights["ln_f_b"])

        return model
