# simple-llm

A simple LLM built with three backends:

- pure python
- nympy
- wgpu-py/vulkan

## Installation and Usage

* Install uv (https://docs.astral.sh/uv/#installation)
* Clone this repo
* `cd simple-llm`
* `uv sync --upgrade`
* `uv run main.py -h`

## BPE Tokenizer

* The bpe\_tokenizer\_slow.py file contains a *pure* python version which is *very* slow.
* Use the default bpe\_tokenizer\_fast.py file after compiling the C version (bpe\_tokenizer.c) with `gcc -O2 -o bpe_tokenizer bpe_tokenizer.c`
