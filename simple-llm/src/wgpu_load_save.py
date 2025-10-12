def save_model(model: TransformerModel, base_path: str) -> None:
    """Save model to checkpoint in backend-agnostic format"""
    backend = model.backend
    params = model.params
    opt_state = model.opt_state

    # Download model parameters
    weights_data = {
        "embedding": _download_buffer_from_gpu_2d(backend, params.embedding),
        "pos_encoding": _download_buffer_from_gpu_2d(backend, params.pos_encoding)
        if params.pos_encoding
        else None,
        "layers": [],
    }

    # Download transformer layers
    for layer in params.layers:
        layer_data = {
            "attn_wq": _download_buffer_from_gpu_2d(backend, layer.attn_wq),
            "attn_wk": _download_buffer_from_gpu_2d(backend, layer.attn_wk),
            "attn_wv": _download_buffer_from_gpu_2d(backend, layer.attn_wv),
            "attn_wo": _download_buffer_from_gpu_2d(backend, layer.attn_wo),
            "ff_w1": _download_buffer_from_gpu_2d(backend, layer.ff_w1),
            "ff_b1": _download_buffer_from_gpu_1d(backend, layer.ff_b1),
            "ff_w2": _download_buffer_from_gpu_2d(backend, layer.ff_w2),
            "ff_b2": _download_buffer_from_gpu_1d(backend, layer.ff_b2),
            "ln_gamma1": _download_buffer_from_gpu_1d(backend, layer.ln_gamma1),
            "ln_beta1": _download_buffer_from_gpu_1d(backend, layer.ln_beta1),
            "ln_gamma2": _download_buffer_from_gpu_1d(backend, layer.ln_gamma2),
            "ln_beta2": _download_buffer_from_gpu_1d(backend, layer.ln_beta2),
        }
        weights_data["layers"].append(layer_data)

    # Download optimizer state in backend-agnostic format
    # Store as nested dict structure that matches Optax's pytree
    optimizer_data = {
        "step": opt_state.step,  # Store step separately for compatibility
        # Momentum (first moment)
        "m": {
            "embedding": _download_buffer_from_gpu_2d(backend, opt_state.m_embedding),
            "layers": [],
        },
        # Variance (second moment)
        "v": {
            "embedding": _download_buffer_from_gpu_2d(backend, opt_state.v_embedding),
            "layers": [],
        },
    }

    # Download momentum for each layer
    for m_layer in opt_state.m_layers:
        m_layer_data = {
            "attn_wq": _download_buffer_from_gpu_2d(backend, m_layer.attn_wq),
            "attn_wk": _download_buffer_from_gpu_2d(backend, m_layer.attn_wk),
            "attn_wv": _download_buffer_from_gpu_2d(backend, m_layer.attn_wv),
            "attn_wo": _download_buffer_from_gpu_2d(backend, m_layer.attn_wo),
            "ff_w1": _download_buffer_from_gpu_2d(backend, m_layer.ff_w1),
            "ff_b1": _download_buffer_from_gpu_1d(backend, m_layer.ff_b1),
            "ff_w2": _download_buffer_from_gpu_2d(backend, m_layer.ff_w2),
            "ff_b2": _download_buffer_from_gpu_1d(backend, m_layer.ff_b2),
            "ln_gamma1": _download_buffer_from_gpu_1d(backend, m_layer.ln_gamma1),
            "ln_beta1": _download_buffer_from_gpu_1d(backend, m_layer.ln_beta1),
            "ln_gamma2": _download_buffer_from_gpu_1d(backend, m_layer.ln_gamma2),
            "ln_beta2": _download_buffer_from_gpu_1d(backend, m_layer.ln_beta2),
        }
        optimizer_data["m"]["layers"].append(m_layer_data)

    # Download variance for each layer
    for v_layer in opt_state.v_layers:
        v_layer_data = {
            "attn_wq": _download_buffer_from_gpu_2d(backend, v_layer.attn_wq),
            "attn_wk": _download_buffer_from_gpu_2d(backend, v_layer.attn_wk),
            "attn_wv": _download_buffer_from_gpu_2d(backend, v_layer.attn_wv),
            "attn_wo": _download_buffer_from_gpu_2d(backend, v_layer.attn_wo),
            "ff_w1": _download_buffer_from_gpu_2d(backend, v_layer.ff_w1),
            "ff_b1": _download_buffer_from_gpu_1d(backend, v_layer.ff_b1),
            "ff_w2": _download_buffer_from_gpu_2d(backend, v_layer.ff_w2),
            "ff_b2": _download_buffer_from_gpu_1d(backend, v_layer.ff_b2),
            "ln_gamma1": _download_buffer_from_gpu_1d(backend, v_layer.ln_gamma1),
            "ln_beta1": _download_buffer_from_gpu_1d(backend, v_layer.ln_beta1),
            "ln_gamma2": _download_buffer_from_gpu_1d(backend, v_layer.ln_gamma2),
            "ln_beta2": _download_buffer_from_gpu_1d(backend, v_layer.ln_beta2),
        }
        optimizer_data["v"]["layers"].append(v_layer_data)

    weights_data["optimizer"] = optimizer_data

    # Save metadata
    metadata = {
        "vocab_size": model.tm_params.vocab_size,
        "embedding_dim": model.tm_params.embedding_dim,
        "context_size": model.tm_params.context_size,
        "n_heads": model.tm_params.n_heads,
        "n_layers": model.tm_params.n_layers,
        "learning_rate": model.learning_rate,
        "total_steps": model.total_steps,
        "epochs": model.tm_params.epochs,
        "backend": "wgpu",  # Mark which backend saved this
    }

    # Write to disk
    model_file, weights_file = get_model_file_names(base_path)
    with open(model_file, "w") as f:
        json.dump(metadata, f, indent=2)

    serialize(weights_data, weights_file)


def load_model(hyper: HyperParams, base_path: str) -> TransformerModel:
    """Load model from checkpoint (works with both backends)"""
    backend = _create_backend_state()

    # Load model metadata
    model_file, weights_file = get_model_file_names(base_path)
    with open(model_file, "r") as f:
        metadata = json.load(f)

    # Load weights from disk
    weights_data = deserialize(weights_file)

    # Upload model parameters
    embedding = _upload_array_to_gpu_2d(backend, weights_data["embedding"])
    pos_encoding = None
    if weights_data.get("pos_encoding") is not None:
        pos_encoding = _upload_array_to_gpu_2d(backend, weights_data["pos_encoding"])

    # Upload transformer layers
    layers = []
    for i in range(metadata["n_layers"]):
        layer_data = weights_data["layers"][i]

        layer = gpu.GPULayerParams(
            attn_wq=_upload_array_to_gpu_2d(backend, layer_data["attn_wq"]),
            attn_wk=_upload_array_to_gpu_2d(backend, layer_data["attn_wk"]),
            attn_wv=_upload_array_to_gpu_2d(backend, layer_data["attn_wv"]),
            attn_wo=_upload_array_to_gpu_2d(backend, layer_data["attn_wo"]),
            ff_w1=_upload_array_to_gpu_2d(backend, layer_data["ff_w1"]),
            ff_b1=_upload_array_to_gpu_1d(backend, layer_data["ff_b1"]),
            ff_w2=_upload_array_to_gpu_2d(backend, layer_data["ff_w2"]),
            ff_b2=_upload_array_to_gpu_1d(backend, layer_data["ff_b2"]),
            ln_gamma1=_upload_array_to_gpu_1d(backend, layer_data["ln_gamma1"]),
            ln_beta1=_upload_array_to_gpu_1d(backend, layer_data["ln_beta1"]),
            ln_gamma2=_upload_array_to_gpu_1d(backend, layer_data["ln_gamma2"]),
            ln_beta2=_upload_array_to_gpu_1d(backend, layer_data["ln_beta2"]),
        )
        layers.append(layer)

    params = gpu.GPUModelParams(
        embedding=embedding, pos_encoding=pos_encoding, layers=layers
    )

    # Load optimizer state (initialize fresh if not available)
    if "optimizer" in weights_data:
        opt_data = weights_data["optimizer"]

        # Upload momentum embedding
        m_embedding = _upload_array_to_gpu_2d(backend, opt_data["m"]["embedding"])
        v_embedding = _upload_array_to_gpu_2d(backend, opt_data["v"]["embedding"])

        # Upload momentum layers
        m_layers = []
        v_layers = []

        for i in range(metadata["n_layers"]):
            m_layer_data = opt_data["m"]["layers"][i]
            v_layer_data = opt_data["v"]["layers"][i]

            m_layer = gpu.GPULayerParams(
                attn_wq=_upload_array_to_gpu_2d(backend, m_layer_data["attn_wq"]),
                attn_wk=_upload_array_to_gpu_2d(backend, m_layer_data["attn_wk"]),
                attn_wv=_upload_array_to_gpu_2d(backend, m_layer_data["attn_wv"]),
                attn_wo=_upload_array_to_gpu_2d(backend, m_layer_data["attn_wo"]),
                ff_w1=_upload_array_to_gpu_2d(backend, m_layer_data["ff_w1"]),
                ff_b1=_upload_array_to_gpu_1d(backend, m_layer_data["ff_b1"]),
                ff_w2=_upload_array_to_gpu_2d(backend, m_layer_data["ff_w2"]),
                ff_b2=_upload_array_to_gpu_1d(backend, m_layer_data["ff_b2"]),
                ln_gamma1=_upload_array_to_gpu_1d(backend, m_layer_data["ln_gamma1"]),
                ln_beta1=_upload_array_to_gpu_1d(backend, m_layer_data["ln_beta1"]),
                ln_gamma2=_upload_array_to_gpu_1d(backend, m_layer_data["ln_gamma2"]),
                ln_beta2=_upload_array_to_gpu_1d(backend, m_layer_data["ln_beta2"]),
            )
            m_layers.append(m_layer)

            v_layer = gpu.GPULayerParams(
                attn_wq=_upload_array_to_gpu_2d(backend, v_layer_data["attn_wq"]),
                attn_wk=_upload_array_to_gpu_2d(backend, v_layer_data["attn_wk"]),
                attn_wv=_upload_array_to_gpu_2d(backend, v_layer_data["attn_wv"]),
                attn_wo=_upload_array_to_gpu_2d(backend, v_layer_data["attn_wo"]),
                ff_w1=_upload_array_to_gpu_2d(backend, v_layer_data["ff_w1"]),
                ff_b1=_upload_array_to_gpu_1d(backend, v_layer_data["ff_b1"]),
                ff_w2=_upload_array_to_gpu_2d(backend, v_layer_data["ff_w2"]),
                ff_b2=_upload_array_to_gpu_1d(backend, v_layer_data["ff_b2"]),
                ln_gamma1=_upload_array_to_gpu_1d(backend, v_layer_data["ln_gamma1"]),
                ln_beta1=_upload_array_to_gpu_1d(backend, v_layer_data["ln_beta1"]),
                ln_gamma2=_upload_array_to_gpu_1d(backend, v_layer_data["ln_gamma2"]),
                ln_beta2=_upload_array_to_gpu_1d(backend, v_layer_data["ln_beta2"]),
            )
            v_layers.append(v_layer)

        opt_state = gpu.GPUOptimizerState(
            m_embedding=m_embedding,
            v_embedding=v_embedding,
            m_layers=m_layers,
            v_layers=v_layers,
            step=opt_data.get("step", 0),
        )
    else:
        # Create fresh optimizer state
        opt_state = _create_optimizer_state(backend, params)

    tm_params = TransformerModelParams(
        vocab_size=metadata["vocab_size"],
        embedding_dim=metadata["embedding_dim"],
        context_size=metadata["context_size"],
        n_heads=metadata["n_heads"],
        n_layers=metadata["n_layers"],
        epochs=metadata.get("epochs", []),
    )

    return TransformerModel(
        tm_params=tm_params,
        params=params,
        opt_state=opt_state,
        learning_rate=metadata.get("learning_rate", 3e-4),
        total_steps=metadata.get("total_steps", 0),
        backend=backend,
    )


# ============================
# # In jax_backend.py
def save_model(model, base_path):
    optimizer_data = {
        "step": model.step,
        "m": jax.tree_util.tree_map(np.array, model.opt_state[0]),  # Convert to numpy
        "v": jax.tree_util.tree_map(np.array, model.opt_state[1]),
    }
    # ... rest of save logic
