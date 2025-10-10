import math
from dataclasses import dataclass


@dataclass
class HyperParams:
    """Hyperparameters for LLM training."""

    embedding_dim: int  # d_model - dimension of embeddings
    n_layers: int  # Number of transformer layers
    n_heads: int  # Number of attention heads
    context_size: int  # Maximum sequence length
    vocab_size: int  # Vocabulary size
    batch_size: int  # Training batch size
    learning_rate: float  # Learning rate

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"HyperParams(\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  n_layers={self.n_layers},\n"
            f"  n_heads={self.n_heads},\n"
            f"  context_size={self.context_size},\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  batch_size={self.batch_size},\n"
            f"  learning_rate={self.learning_rate:.2e}\n"
            f")"
        )

    @property
    def approx_params(self) -> int:
        """Approximate total parameter count."""
        embed_params = self.vocab_size * self.embedding_dim
        layer_params = self.n_layers * (12 * self.embedding_dim**2)
        return embed_params + layer_params


def get_hyperparams_auto(
    hp: HyperParams, char_count: int, chars_per_token: float = 4.0
) -> HyperParams:
    """
    Automatically determine hyperparameters based on corpus character count.

    Uses continuous scaling laws with smooth log-scale interpolation between
    anchor points. This eliminates manual preset selection and provides
    configurations optimally tuned to the exact corpus size.

    Args:
        char_count: Number of characters in the corpus
        chars_per_token: Estimated characters per token (default 4.0 for English)
                        Adjust for other languages: ~3.5 for Romance languages,
                        ~5.0 for German, ~2.0 for Chinese
    """

    estimated_tokens = char_count / chars_per_token

    # Use log10 scale for smooth scaling across orders of magnitude
    log_tokens = math.log10(max(estimated_tokens, 1))

    # Define 15 anchor points based on scaling laws
    # Format: (log10(tokens), d_model, n_layers, n_heads, context, vocab, batch, lr)
    anchors = [
        (2.0, 48, 2, 2, 64, 256, 2, 5e-3),  # 100
        (2.5, 64, 2, 2, 128, 512, 4, 3e-3),  # ~300
        (3.0, 96, 3, 4, 192, 1024, 8, 1.5e-3),  # 1K
        (3.3, 128, 4, 4, 256, 1536, 12, 1.2e-3),  # ~2K
        (3.7, 160, 4, 4, 384, 2048, 18, 9e-4),  # ~5K
        (4.0, 192, 5, 4, 384, 2048, 24, 8e-4),  # 10K
        (4.3, 224, 6, 8, 512, 3072, 36, 6e-4),  # ~20K
        (4.7, 288, 7, 8, 512, 4096, 56, 5e-4),  # ~50K
        (5.0, 384, 8, 8, 768, 8192, 96, 3e-4),  # 100K
        (5.3, 448, 9, 8, 1024, 10240, 128, 2.5e-4),  # ~200K
        (5.7, 576, 10, 12, 1024, 16384, 192, 1.8e-4),  # ~500K
        (6.0, 768, 12, 12, 1536, 24576, 320, 1e-4),  # 1M
        (6.3, 896, 14, 16, 2048, 32768, 416, 8e-5),  # ~2M
        (6.7, 1024, 16, 16, 2048, 50000, 512, 6e-5),  # ~5M
        (7.0, 1280, 20, 16, 2048, 65536, 768, 4e-5),  # 10M
    ]

    # Handle edge cases
    if log_tokens <= anchors[0][0]:
        _, d, l, h, c, v, b, lr = anchors[0]
        return HyperParams(d, l, h, c, v, b, lr)

    if log_tokens >= anchors[-1][0]:
        _, d, l, h, c, v, b, lr = anchors[-1]
        return HyperParams(d, l, h, c, v, b, lr)

    # Find bracketing anchors and interpolate
    for i in range(len(anchors) - 1):
        lower_log, *lower_params = anchors[i]
        upper_log, *upper_params = anchors[i + 1]

        if lower_log <= log_tokens <= upper_log:
            # Linear interpolation factor (0.0 to 1.0)
            t = (log_tokens - lower_log) / (upper_log - lower_log)

            # Unpack parameters
            (d_lower, l_lower, h_lower, c_lower, v_lower, b_lower, lr_lower) = (
                lower_params
            )
            (d_upper, l_upper, h_upper, c_upper, v_upper, b_upper, lr_upper) = (
                upper_params
            )

            # Linear interpolation for most parameters
            embedding_dim = round(d_lower + t * (d_upper - d_lower))
            n_layers = round(l_lower + t * (l_upper - l_lower))
            n_heads = round(h_lower + t * (h_upper - h_lower))
            context_size = round(c_lower + t * (c_upper - c_lower))
            vocab_size = round(v_lower + t * (v_upper - v_lower))
            batch_size = round(b_lower + t * (b_upper - b_lower))

            # Geometric (log-scale) interpolation for learning rate
            lr_log = math.log(lr_lower) + t * (math.log(lr_upper) - math.log(lr_lower))
            learning_rate = math.exp(lr_log)

            # Ensure n_heads divides embedding_dim evenly
            if embedding_dim % n_heads != 0:
                divisors = [
                    h
                    for h in range(1, min(17, embedding_dim + 1))
                    if embedding_dim % h == 0
                ]
                if divisors:
                    n_heads = min(divisors, key=lambda x: abs(x - n_heads))
                else:
                    embedding_dim = n_heads * (embedding_dim // n_heads)

            # Round context and vocab to powers of 2 (hardware efficient)
            context_size = 2 ** round(math.log2(max(context_size, 32)))
            vocab_size = 2 ** round(math.log2(max(vocab_size, 256)))
            batch_size = max(1, batch_size)

            batch_size = hp.batch_size or batch_size
            base_lr = hp.learning_rate or learning_rate
            learning_rate = base_lr * math.sqrt(batch_size / 16)
            return HyperParams(
                embedding_dim=hp.embedding_dim or embedding_dim,
                n_layers=hp.n_layers or n_layers,
                n_heads=hp.n_heads or n_heads,
                context_size=hp.context_size or context_size,
                vocab_size=hp.vocab_size or vocab_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

    raise NotImplementedError()
