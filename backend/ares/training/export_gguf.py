"""
ARES Phase 4: PEFT LoRA Adapter to GGUF Converter.
Converts trained PEFT LoRA safetensors (448 tensors across 32 layers)
into a standard GGUF LoRA adapter that Ollama loads with `ADAPTER <path>.gguf`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import numpy as np
import torch
from safetensors.torch import load_file
import gguf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard PEFT to llama.cpp tensor naming map
TENSOR_MAP = {
    "self_attn.q_proj": "attn_q.weight",
    "self_attn.k_proj": "attn_k.weight",
    "self_attn.v_proj": "attn_v.weight",
    "self_attn.o_proj": "attn_output.weight",
    "mlp.gate_proj": "ffn_gate.weight",
    "mlp.up_proj": "ffn_up.weight",
    "mlp.down_proj": "ffn_down.weight",
}


def convert_lora_to_gguf(
    adapter_dir: Path,
    output_gguf_path: Path,
    base_model_architecture: str = "llama",
):
    """
    Converts a PEFT adapter directory containing adapter_model.safetensors
    and adapter_config.json into a single GGUF LoRA adapter.
    """
    adapter_dir = Path(adapter_dir)
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"

    if not safetensors_path.exists():
        raise FileNotFoundError(f"Missing {safetensors_path}")

    # Load adapter config
    lora_alpha = 32.0
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            lora_alpha = float(cfg.get("lora_alpha", 32.0))

    logger.info("Loading safetensors from %s...", safetensors_path)
    weights = load_file(str(safetensors_path))
    logger.info("Loaded %d tensors from adapter (LoRA alpha=%.1f)", len(weights), lora_alpha)

    # Initialize GGUF Writer
    writer = gguf.GGUFWriter(path=output_gguf_path, arch=base_model_architecture)
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", lora_alpha)

    # Regex to match: base_model.model.model.layers.<layer_idx>.<submodule>.lora_<A|B>.weight
    pattern = re.compile(r"layers\.(\d+)\.(.+?)\.lora_([AB])\.weight")

    converted_count = 0
    for key, tensor in weights.items():
        m = pattern.search(key)
        if not m:
            continue
        layer_idx, module_name, ab = m.groups()
        mapped_suffix = TENSOR_MAP.get(module_name)
        if not mapped_suffix:
            continue

        gguf_name = f"blk.{layer_idx}.{mapped_suffix}.lora{ab}"
        arr = tensor.detach().cpu().to(torch.float32).numpy()

        writer.add_tensor(gguf_name, arr)
        converted_count += 1

    logger.info("Writing %d mapped tensors to %s...", converted_count, output_gguf_path)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = output_gguf_path.stat().st_size / (1024 * 1024)
    logger.info("=== Successfully exported GGUF LoRA adapter: %s (%.2f MB) ===", output_gguf_path, size_mb)
    return output_gguf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PEFT LoRA safetensors to GGUF")
    parser.add_argument("--adapter-dir", "--adapter-path", dest="adapter_dir", type=str, required=True, help="Directory containing adapter_model.safetensors")
    parser.add_argument("--output", "--output-path", dest="output", type=str, required=True, help="Target GGUF file path")
    parser.add_argument("--base-model", type=str, default=None, help="Optional base model path")
    args, _ = parser.parse_known_args()

    convert_lora_to_gguf(Path(args.adapter_dir), Path(args.output))
