from common.util import load_tokenizer
from v1.jax_backend import load_model_config, model_init, model_load
from v1.jax_backend_generate import main_generate


def generate_next(
    tokenizer, tokenizer_path: str, params, config, temperature, prompt: str
):
    xs = tokenizer.encode(prompt, tokenizer_path)
    max_prompt_len = min(len(xs), 20)
    xs = xs[:max_prompt_len]

    ys = main_generate(
        params=params,
        config=config,
        xs=xs,
        max_length=min(10, config.max_seq_len),
    )

    return tokenizer.decode(ys)


def generate_during_training(
    config, params, tokenizer, tokenizer_path, step_count, temperature=0.8
):
    """Generate a sample during training."""
    prompts = [
        "when ",
        "he ",
        "I ",
        "the ",
        " of",
    ]

    # Use a different prompt each time
    import random

    prompt = prompts[random.randint(0, len(prompts) - 1)]
    generated_text = generate_next(
        tokenizer, tokenizer_path, params, config, temperature, prompt
    )

    return f"\nStep {step_count} | Prompt: '{prompt}' → '{generated_text}'\n"


def generate_command(args):
    tokenizer, tokenizer_path = load_tokenizer(args.model_path)
    model_config = load_model_config(args.model_path, tokenizer.vocab_size, args)
    x_model = model_init(model_config)

    # Load trained parameters
    params = model_load(args.model_path)
    if params is None:
        print("No trained model found!")
        return

    print("Model loaded.")

    generated_text = generate_next(
        tokenizer, tokenizer_path, params, x_model.model, 0.8, args.prompt
    )

    print(f"Generated: {generated_text}")
