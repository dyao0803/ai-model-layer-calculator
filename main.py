#!/usr/bin/env python3
"""
Estimate how many layers of a GGUF model to offload to the GPU in LM Studio.

Usage:
    python llm_layers.py 12B Q4 --ram=32gb --vram=10gb
    python llm_layers.py 24b q5_k_m --ram=32gb --vram=10gb --ctx=16384
    python llm_layers.py 32B Q4 --ram=64gb --vram=12gb --layers=64

Everything is case-insensitive. --ram and --vram are required.
"""

import argparse
import math
import re
import sys

# Approximate effective bits per weight for common GGUF quants.
BITS_PER_WEIGHT = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.6,
    "Q5_K_M": 5.7, "Q5_K_S": 5.5, "Q5_0": 5.5, "Q5_1": 6.0,
    "Q4_K_M": 4.85, "Q4_K_S": 4.6, "Q4_0": 4.5, "Q4_1": 5.0,
    "Q3_K_L": 4.3, "Q3_K_M": 3.9, "Q3_K_S": 3.5,
    "Q2_K": 2.6,
    "IQ4_XS": 4.3, "IQ3_M": 3.7, "IQ3_XS": 3.3, "IQ2_M": 2.7, "IQ2_XS": 2.3,
}

# Bare "q4" / "q5" etc. map to the usual default variant.
QUANT_ALIASES = {
    "Q8": "Q8_0", "Q6": "Q6_K", "Q5": "Q5_K_M", "Q4": "Q4_K_M",
    "Q3": "Q3_K_M", "Q2": "Q2_K", "FP16": "F16", "FP32": "F32",
    "IQ4": "IQ4_XS", "IQ3": "IQ3_M", "IQ2": "IQ2_M",
}

# Rough layer counts by parameter size (billions). Real value depends on the
# architecture, so use --layers to override when you know it.
LAYER_TABLE = [
    (1.5, 28), (3.5, 32), (5, 32), (8.5, 32), (10, 36), (13, 40),
    (16, 40), (20, 44), (25, 40), (28, 46), (34, 60), (40, 64),
    (50, 64), (75, 80), (110, 88), (200, 96), (1000, 126),
]


def parse_params(text: str) -> float:
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([bm])?\s*", text.lower())
    if not m:
        sys.exit(f"Can't read parameter size '{text}'. Try 12B, 7b, 0.5B, 350M.")
    value = float(m.group(1))
    return value / 1000 if m.group(2) == "m" else value


def parse_quant(text: str) -> str:
    q = re.sub(r"[-\s.]", "_", text.upper()).strip("_")
    q = q.replace("QK", "Q_K")
    if q in QUANT_ALIASES:
        return QUANT_ALIASES[q]
    if q in BITS_PER_WEIGHT:
        return q
    # allow "Q4KM" -> Q4_K_M
    m = re.fullmatch(r"(I?Q\d)K([MSL])", q)
    if m:
        q = f"{m.group(1)}_K_{m.group(2)}"
        if q in BITS_PER_WEIGHT:
            return q
    m = re.fullmatch(r"(I?Q\d)_?K", q)
    if m and f"{m.group(1)}_K" in BITS_PER_WEIGHT:
        return f"{m.group(1)}_K"
    sys.exit(f"Unknown quantization '{text}'. Known: {', '.join(BITS_PER_WEIGHT)}")


def parse_mem_gb(text: str) -> float:
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*(gb|g|mb|m|tb|t)?\s*", text.lower())
    if not m:
        sys.exit(f"Can't read memory size '{text}'. Try 32gb, 10GB, 8192mb.")
    value, unit = float(m.group(1)), m.group(2) or "gb"
    return {"gb": value, "g": value, "mb": value / 1024, "m": value / 1024,
            "tb": value * 1024, "t": value * 1024}[unit]


def estimate_layers(params_b: float) -> int:
    for limit, layers in LAYER_TABLE:
        if params_b <= limit:
            return layers
    return LAYER_TABLE[-1][1]


def main() -> None:
    p = argparse.ArgumentParser(description="Suggest LM Studio GPU offload layers.")
    p.add_argument("params", help="model size, e.g. 12B")
    p.add_argument("quant", help="quantization, e.g. Q4, Q5_K_M, q8")
    p.add_argument("--ram", required=True, help="system RAM, e.g. 32gb")
    p.add_argument("--vram", required=True, help="GPU VRAM, e.g. 10gb")
    p.add_argument("--ctx", type=int, default=8192, help="context length (default 8192)")
    p.add_argument("--layers", type=int, help="override total layer count")
    p.add_argument("--kv", default="f16", type=str.lower, choices=["f16", "q8", "q4"],
                   help="KV cache type (default f16)")
    p.add_argument("--kv-dim", type=int, default=1024,
                   help="n_kv_heads * head_dim (default 1024, typical GQA model)")
    p.add_argument("--overhead", type=float, default=1.0,
                   help="GB of VRAM reserved for compute buffers/display (default 1.0)")
    p.add_argument("--os-reserve", type=float, default=4.0,
                   help="GB of RAM reserved for the OS (default 4.0)")
    args = p.parse_args()

    params_b = parse_params(args.params)
    quant = parse_quant(args.quant)
    ram = parse_mem_gb(args.ram)
    vram = parse_mem_gb(args.vram)

    bpw = BITS_PER_WEIGHT[quant]
    total_layers = args.layers or estimate_layers(params_b)
    layers_known = args.layers is not None

    # Model file size.
    model_gb = params_b * 1e9 * bpw / 8 / 1024**3
    # Embeddings + output head sit outside the repeating blocks (~5% of weights).
    non_layer_gb = model_gb * 0.05
    per_layer_gb = (model_gb - non_layer_gb) / total_layers

    # KV cache per layer: K and V, each kv_dim wide, per token.
    kv_bytes = {"f16": 2.0, "q8": 1.0625, "q4": 0.5625}[args.kv]
    kv_per_layer_gb = 2 * args.kv_dim * kv_bytes * args.ctx / 1024**3

    cost_per_layer = per_layer_gb + kv_per_layer_gb
    usable_vram = vram - args.overhead

    # Reserve the output head on GPU first, then fit as many blocks as possible.
    budget = usable_vram - non_layer_gb
    if budget <= 0:
        gpu_layers = 0
    else:
        gpu_layers = min(total_layers, math.floor(budget / cost_per_layer))

    gpu_gb = gpu_layers * cost_per_layer + (non_layer_gb if gpu_layers > 0 else 0)
    cpu_layers = total_layers - gpu_layers
    cpu_gb = cpu_layers * cost_per_layer + (non_layer_gb if gpu_layers == 0 else 0)
    ram_avail = ram - args.os_reserve

    print(f"Model:        {params_b:g}B @ {quant} ({bpw} bits/weight)")
    print(f"Hardware:     {ram:g} GB RAM, {vram:g} GB VRAM")
    print(f"Context:      {args.ctx} tokens, KV cache {args.kv}")
    print(f"Layers:       {total_layers} "
          f"({'user supplied' if layers_known else 'estimated from size, use --layers to override'})")
    print()
    print(f"Weights size:       ~{model_gb:.1f} GB")
    print(f"Per layer:          ~{per_layer_gb * 1024:.0f} MB weights "
          f"+ ~{kv_per_layer_gb * 1024:.0f} MB KV cache")
    print(f"Usable VRAM:        ~{usable_vram:.1f} GB (after {args.overhead:g} GB overhead)")
    print()
    if gpu_layers >= total_layers:
        print(f">>> LM Studio GPU Offload: {total_layers} (max, fully on GPU, ~{gpu_gb:.1f} GB VRAM)")
    elif gpu_layers == 0:
        print(">>> LM Studio GPU Offload: 0 (not even one layer fits, CPU only)")
    else:
        print(f">>> LM Studio GPU Offload: {gpu_layers} of {total_layers} "
              f"(~{gpu_gb:.1f} GB VRAM, {cpu_layers} layers on CPU)")
    print(f"    CPU/RAM side: ~{cpu_gb:.1f} GB of {ram_avail:.1f} GB available")

    if cpu_gb > ram_avail:
        print("\nWARNING: the CPU-side portion exceeds available RAM. "
              "Use a smaller quant, a shorter context, or a smaller model.")
    elif gpu_layers < total_layers:
        print("\nTip: lower --ctx or use --kv=q8 to free VRAM for more layers.")
    if gpu_layers > 0 and gpu_layers < total_layers:
        print("Tip: if LM Studio runs out of memory, drop the value by 1-2 layers.")


if __name__ == "__main__":
    main()
