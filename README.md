# ai-model-layer-calculator

A small CLI tool that estimates how many layers of a GGUF model to offload
to your GPU, so you have a concrete number to type into **LM Studio**'s
"GPU Offload" slider instead of guessing.

Given a model size, quantization, and your RAM/VRAM, it works out:

- how big the model's weights are on disk,
- how much VRAM each transformer layer plus its KV cache costs at your
  chosen context length,
- how many layers fit in your GPU before you run out of VRAM,
- whether the remaining CPU-side layers fit in your available RAM.

## Requirements

- Python 3.8+
- No third-party dependencies (standard library only)

## Usage

```bash
python main.py <params> <quant> --ram=<size> --vram=<size> [options]
```

`--ram` and `--vram` are required — there are no defaults, since the right
value is entirely dependent on your machine.

### Examples

```bash
python main.py 12B Q4 --ram=32gb --vram=10gb
python main.py 24b q5_k_m --ram=32gb --vram=10gb --ctx=16384
python main.py 32B Q4 --ram=64gb --vram=12gb --layers=64
```

Everything is case-insensitive: `12B`/`12b`, `Q4`/`q4`, `32GB`/`32gb` all work.

### Arguments

| Argument   | Required | Description                                                        |
|------------|----------|----------------------------------------------------------------------|
| `params`   | yes      | Model parameter count, e.g. `12B`, `7b`, `0.5B`, `350M`             |
| `quant`    | yes      | Quantization, e.g. `Q4`, `Q4_K_M`, `Q5_K_M`, `Q8_0`, `IQ3_M`         |
| `--ram`    | yes      | System RAM, e.g. `32gb`, `16GB`, `32768mb`                          |
| `--vram`   | yes      | GPU VRAM, e.g. `10gb`, `24GB`                                       |

### Options

| Option          | Default | Description                                                   |
|------------------|---------|-----------------------------------------------------------------|
| `--ctx`          | `8192`  | Context length in tokens                                      |
| `--layers`       | estimated from model size | Override the model's total layer count, if you know it |
| `--kv`           | `f16`   | KV cache type: `f16`, `q8`, or `q4`                            |
| `--kv-dim`       | `1024`  | `n_kv_heads * head_dim` for the model (typical GQA model)      |
| `--overhead`     | `1.0`   | GB of VRAM reserved for compute buffers / display              |
| `--os-reserve`   | `4.0`   | GB of RAM reserved for the OS                                  |

## Example output

```
$ python main.py 12B Q4 --ram=32gb --vram=10gb

Model:        12B @ Q4_K_M (4.85 bits/weight)
Hardware:     32 GB RAM, 10 GB VRAM
Context:      8192 tokens, KV cache f16
Layers:       40 (estimated from size, use --layers to override)

Weights size:       ~6.8 GB
Per layer:          ~165 MB weights + ~32 MB KV cache
Usable VRAM:        ~9.0 GB (after 1 GB overhead)

>>> LM Studio GPU Offload: 40 (max, fully on GPU, ~8.0 GB VRAM)
    CPU/RAM side: ~0.0 GB of 28.0 GB available
```

The `>>> LM Studio GPU Offload: N` line is the number to enter into LM
Studio's GPU offload setting for that model.

## Notes on accuracy

- **Layer counts** are looked up from a rough table keyed by parameter
  size, since the real value depends on the model's architecture. Pass
  `--layers` if you know the exact figure (e.g. from the model card or
  `llama.cpp`'s loader output) for a more accurate estimate.
- **Bits-per-weight** values for each quantization are approximate
  averages — actual GGUF file sizes vary slightly by model.
- The tool assumes the embedding and output head weights are ~5% of the
  total model size and, when offloaded, sit on the GPU alongside whatever
  transformer layers fit.
- If the CPU-side portion exceeds your available RAM, the tool warns you —
  try a smaller quant, a shorter `--ctx`, or `--kv=q8`/`--kv=q4` to shrink
  the KV cache.

## License

MIT
