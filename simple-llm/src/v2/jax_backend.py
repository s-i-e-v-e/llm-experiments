import dataclasses
import json
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import struct

from common.util import deserialize, get_model_file_names, serialize

jax.config.update("jax_enable_x64", False)


# Positional encoding utility
def positional_encoding(seq_len, dim):
    pos = jnp.arange(seq_len)[:, None]
    i = jnp.arange(dim)[None, :]
    angle_rates = 1 / (10000 ** (2 * (i // 2) / dim))
    angle_rads = pos * angle_rates
    sines = jnp.sin(angle_rads[:, 0::2])
    cosines = jnp.cos(angle_rads[:, 1::2])
    pos_encoding = jnp.concatenate([sines, cosines], axis=-1)
    return pos_encoding


@struct.dataclass
class ModelParams:
    embedding: jnp.ndarray
    pos_encoding: jnp.ndarray
    attn_wq: jnp.ndarray
    attn_wk: jnp.ndarray
    attn_wv: jnp.ndarray
    attn_wo: jnp.ndarray
    ff_w1: jnp.ndarray
    ff_b1: jnp.ndarray
    ff_w2: jnp.ndarray
    ff_b2: jnp.ndarray
    ln_gamma1: jnp.ndarray
    ln_beta1: jnp.ndarray
    ln_gamma2: jnp.ndarray
    ln_beta2: jnp.ndarray


@dataclasses.dataclass
class TransformerModelParams:
    vocab_size: int
    embedding_dim: int
    context_length: int
    n_heads: int
    n_layers: int


@dataclasses.dataclass
class TransformerModel:
    tm_params: TransformerModelParams
    params: ModelParams
    opt_state: optax.OptState
    optimizer: optax.GradientTransformation


def save_model(model: TransformerModel, model_path: str):
    xs = get_model_file_names(model_path)

    serialize(model.params, xs[0])
    serialize(model.opt_state, xs[1])

    with open(xs[2], "w") as f:
        json.dump(dataclasses.asdict(model.tm_params), f)


def load_model(learning_rate: float, model_path: str):
    xs = get_model_file_names(model_path)

    params = ModelParams(**deserialize(None, xs[0]))
    ox = optax.adam(learning_rate=learning_rate)
    opt_state = deserialize(ox.init(params), xs[1])

    with open(xs[2], "r") as f:
        tm_params = TransformerModelParams(**json.load(f))

    return TransformerModel(
        tm_params=tm_params,
        params=params,
        opt_state=opt_state,
        optimizer=optax.adam(learning_rate=learning_rate),
    )


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    norm = (x - mean) / jnp.sqrt(var + eps)
    return norm * gamma + beta


def split_heads(x, n_heads, head_dim):
    B, T, C = x.shape
    x = x.reshape(B, T, n_heads, head_dim)
    return x.transpose(0, 2, 1, 3)  # (B, heads, T, head_dim)


def combine_heads(x):
    B, H, T, D = x.shape
    x = x.transpose(0, 2, 1, 3).reshape(B, T, H * D)
    return x


def scaled_dot_product_attention(q, k, v):
    matmul_qk = jnp.matmul(q, jnp.swapaxes(k, -1, -2))
    scale = q.shape[-1] ** 0.5
    logits = matmul_qk / scale
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.matmul(weights, v)


def multi_head_attention(x, params, n_heads, head_dim):
    q = jnp.dot(x, params.attn_wq)
    k = jnp.dot(x, params.attn_wk)
    v = jnp.dot(x, params.attn_wv)

    q = split_heads(q, n_heads, head_dim)
    k = split_heads(k, n_heads, head_dim)
    v = split_heads(v, n_heads, head_dim)

    attn_output = scaled_dot_product_attention(q, k, v)
    attn_output = combine_heads(attn_output)
    attn_output = jnp.dot(attn_output, params.attn_wo)
    return attn_output


def feed_forward(x, params):
    ff = jnp.dot(x, params.ff_w1) + params.ff_b1
    ff = jax.nn.relu(ff)
    ff = jnp.dot(ff, params.ff_w2) + params.ff_b2
    return ff


def transformer_block(x, params, n_heads, head_dim):
    attn_output = multi_head_attention(x, params, n_heads, head_dim)
    x = layer_norm(x + attn_output, params.ln_gamma1, params.ln_beta1)

    ff_output = feed_forward(x, params)
    out = layer_norm(x + ff_output, params.ln_gamma2, params.ln_beta2)
    return out


def forward(params, inputs, n_layers, n_heads, head_dim):
    x = params.embedding[inputs] + params.pos_encoding[None, :, :]
    for _ in range(n_layers):
        x = transformer_block(x, params, n_heads, head_dim)
    logits = jnp.dot(x, params.embedding.T)
    return logits


def loss_fn(
    params, batch_inputs, batch_targets, vocab_size, n_layers, n_heads, head_dim
):
    logits = forward(params, batch_inputs, n_layers, n_heads, head_dim)
    log_probs = jax.nn.log_softmax(logits)
    one_hot_targets = jax.nn.one_hot(batch_targets, vocab_size)
    loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
    return jnp.mean(loss)


@partial(
    jax.jit,
    static_argnames=["vocab_size", "n_layers", "n_heads", "head_dim", "optimizer"],
)
def train_step(
    params,
    opt_state,
    batch_inputs,
    batch_targets,
    vocab_size,
    n_layers,
    n_heads,
    head_dim,
    optimizer,
):
    def internal_loss_fn(p):
        return loss_fn(
            p, batch_inputs, batch_targets, vocab_size, n_layers, n_heads, head_dim
        )

    loss, grads = jax.value_and_grad(internal_loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


def update_params(model: TransformerModel, batch_inputs, batch_targets):
    new_params, new_opt_state, loss = train_step(
        model.params,
        model.opt_state,
        batch_inputs,
        batch_targets,
        vocab_size=model.tm_params.vocab_size,
        n_layers=model.tm_params.n_layers,
        n_heads=model.tm_params.n_heads,
        head_dim=model.tm_params.embedding_dim // model.tm_params.n_heads,
        optimizer=model.optimizer,
    )
    return TransformerModel(
        TransformerModelParams(
            vocab_size=model.tm_params.vocab_size,
            embedding_dim=model.tm_params.embedding_dim,
            context_length=model.tm_params.context_length,
            n_heads=model.tm_params.n_heads,
            n_layers=model.tm_params.n_layers,
        ),
        params=new_params,
        opt_state=new_opt_state,
        optimizer=model.optimizer,
    ), loss


def to_backend(data):
    return jnp.array(data)


def initialize_model(
    vocab_size,
    embedding_dim,
    context_length,
    n_heads,
    n_layers,
    learning_rate=0.001,
):
    key = jax.random.PRNGKey(0)
    head_dim = embedding_dim // n_heads
    k1, k2, *ks = jax.random.split(key, 13)
    params = ModelParams(
        embedding=jax.random.normal(k1, (vocab_size, embedding_dim)) * 0.01,
        pos_encoding=positional_encoding(context_length, embedding_dim),
        attn_wq=jax.random.normal(k2, (embedding_dim, embedding_dim)) * 0.01,
        attn_wk=jax.random.normal(ks[0], (embedding_dim, embedding_dim)) * 0.01,
        attn_wv=jax.random.normal(ks[1], (embedding_dim, embedding_dim)) * 0.01,
        attn_wo=jax.random.normal(ks[2], (embedding_dim, embedding_dim)) * 0.01,
        ff_w1=jax.random.normal(ks[3], (embedding_dim, 4 * embedding_dim)) * 0.01,
        ff_b1=jnp.zeros((4 * embedding_dim,)),
        ff_w2=jax.random.normal(ks[4], (4 * embedding_dim, embedding_dim)) * 0.01,
        ff_b2=jnp.zeros((embedding_dim,)),
        ln_gamma1=jnp.ones((embedding_dim,)),
        ln_beta1=jnp.zeros((embedding_dim,)),
        ln_gamma2=jnp.ones((embedding_dim,)),
        ln_beta2=jnp.zeros((embedding_dim,)),
    )
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    return TransformerModel(
        TransformerModelParams(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            context_length=context_length,
            n_heads=n_heads,
            n_layers=n_layers,
        ),
        params=params,
        opt_state=opt_state,
        optimizer=optimizer,
    )
