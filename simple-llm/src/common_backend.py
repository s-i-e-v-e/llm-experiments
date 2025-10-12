import dataclasses


@dataclasses.dataclass
class TransformerModelParams:
    vocab_size: int
    embedding_dim: int
    context_size: int
    n_heads: int
    n_layers: int
    epochs: list = dataclasses.field(default_factory=list)
