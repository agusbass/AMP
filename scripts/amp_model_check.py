#!/usr/bin/env python3
"""amp_model_check.py - does AMP's FlashAttention-2 kernel actually support
this open-source model's attention shape?

This answers a question Track 3 of the hackathon explicitly asks for ("use
any open-source model + AMD infrastructure") that the rest of AMP's tooling
doesn't: given a real model's published architecture, is AMP's kernel even
able to serve it correctly?

Two hard constraints come directly from kernels/flash_attn.cuh and the GQA
head-grouping arithmetic in kernels/flash_attn.cu (`hkv = hq / (H_q / H_kv)`
at flash_attn.cu:90 - integer division, so it silently misgroups KV heads,
not crashes, if H_q is not a clean multiple of H_kv):

  1. head_dim must be a multiple of 16 and <= 256 (kernels/flash_attn.cuh)
  2. num_attention_heads must be evenly divisible by num_key_value_heads
     (flash_attn.cu:90 - GQA grouping correctness, not just a style rule)

This is a static check against published numbers - it does NOT run the
kernel and does NOT require a GPU or compiler. It tells you whether AMP's
kernel is even a candidate for this model's attention layer; it does not
replace amp_verify_matmul / parity_check.py for actually proving the port
correct once you do have hardware.

The built-in model configs below were fetched from each model's public,
non-gated config.json on Hugging Face (see source URLs in MODEL_CONFIGS) -
not reproduced from memory. head_dim is not an explicit config.json field
for any of these models; it is derived as hidden_size // num_attention_heads,
which is the standard convention these models' own modeling code uses
(confirm against the model's actual modeling_*.py if it ever overrides this).

Usage:
    python3 scripts/amp_model_check.py --list
    python3 scripts/amp_model_check.py mistral-7b-v0.1
    python3 scripts/amp_model_check.py --custom --hidden-size 4096 --heads 32 --kv-heads 7
    python3 scripts/amp_model_check.py mistral-7b-v0.1 --diagnose
"""
import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

# Fetched from the listed Hugging Face config.json URLs (non-gated repos),
# not reproduced from training-data memory. Re-fetch before relying on this
# for an actual migration decision - models get updated.
MODEL_CONFIGS = {
    "mistral-7b-v0.1": {
        "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
        "intermediate_size": 14336, "max_position_embeddings": 32768,
        "source": "https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json",
    },
    "tinyllama-1.1b-chat-v1.0": {
        "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 4,
        "intermediate_size": 5632, "max_position_embeddings": 2048,
        "source": "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/config.json",
    },
    "qwen2-7b": {
        "hidden_size": 3584, "num_attention_heads": 28, "num_key_value_heads": 4,
        "intermediate_size": 18944, "max_position_embeddings": 131072,
        "source": "https://huggingface.co/Qwen/Qwen2-7B/raw/main/config.json",
    },
    "phi-3-mini-4k-instruct": {
        "hidden_size": 3072, "num_attention_heads": 32, "num_key_value_heads": 32,
        "intermediate_size": 8192, "max_position_embeddings": 4096,
        "source": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/raw/main/config.json",
    },
    # Not a real model - included only to demonstrate what a FAIL looks like.
    # 30 attention heads grouped into 7 kv heads does not divide evenly.
    "example-bad-gqa-fictional": {
        "hidden_size": 3840, "num_attention_heads": 30, "num_key_value_heads": 7,
        "intermediate_size": 10240, "max_position_embeddings": 4096,
        "source": "(fictional - hand-picked to fail the GQA divisibility check, not a real model)",
    },
}


def check_model(name, cfg):
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    head_dim = hidden // heads
    head_dim_exact = hidden % heads == 0

    checks = []
    checks.append(("head_dim is a whole number (hidden_size divisible by num_attention_heads)",
                    head_dim_exact,
                    f"hidden_size={hidden} / num_attention_heads={heads} = {hidden / heads}"))
    checks.append((f"head_dim={head_dim} is a multiple of 16 (kernels/flash_attn.cuh)",
                    head_dim % 16 == 0, None))
    checks.append((f"head_dim={head_dim} <= 256 (kernels/flash_attn.cuh)",
                    head_dim <= 256, None))
    checks.append((f"num_key_value_heads={kv_heads} <= num_attention_heads={heads}",
                    kv_heads <= heads, None))
    checks.append((f"num_attention_heads={heads} evenly divisible by num_key_value_heads={kv_heads} "
                    f"(flash_attn.cu:90 GQA grouping: hkv = hq / (H_q / H_kv))",
                    heads % kv_heads == 0,
                    None if heads % kv_heads == 0 else
                    f"{heads}/{kv_heads} = {heads/kv_heads} -- integer division would silently "
                    f"misgroup KV heads, not crash"))
    return head_dim, checks


def print_report(name, cfg, head_dim, checks):
    print(f"Model: {name}")
    print(f"  hidden_size={cfg['hidden_size']} num_attention_heads={cfg['num_attention_heads']} "
          f"num_key_value_heads={cfg['num_key_value_heads']} (derived head_dim={head_dim})")
    print(f"  source: {cfg['source']}")
    print()
    all_ok = True
    for desc, ok, extra in checks:
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {desc}")
        if extra:
            print(f"         {extra}")
    print()
    if all_ok:
        print(f"Result: COMPATIBLE -- AMP's FlashAttention-2 kernel can serve "
              f"{name}'s attention shapes as written.")
    else:
        print(f"Result: NOT COMPATIBLE AS-IS -- see FAIL line(s) above.")
    print()
    print("This only checks the constraints above against published config "
          "numbers; it does not run the kernel. Use amp_verify_matmul / "
          "parity_check.py on real hardware to actually prove a port correct.")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", choices=sorted(MODEL_CONFIGS), help="model to check")
    ap.add_argument("--list", action="store_true", help="list built-in model configs")
    ap.add_argument("--custom", action="store_true", help="check a custom config via flags below")
    ap.add_argument("--hidden-size", type=int)
    ap.add_argument("--heads", type=int)
    ap.add_argument("--kv-heads", type=int)
    ap.add_argument("--diagnose", action="store_true",
                     help="also run amp_diagnose.py against kernels/flash_attn.cu(.cuh)")
    args = ap.parse_args()

    if args.list:
        for name, cfg in sorted(MODEL_CONFIGS.items()):
            print(f"  {name:<28} hidden={cfg['hidden_size']:<6} heads={cfg['num_attention_heads']:<3} "
                  f"kv_heads={cfg['num_key_value_heads']}")
        return

    if args.custom:
        if args.hidden_size is None or args.heads is None or args.kv_heads is None:
            ap.error("--custom requires --hidden-size, --heads, and --kv-heads")
        name = "custom"
        cfg = {"hidden_size": args.hidden_size, "num_attention_heads": args.heads,
               "num_key_value_heads": args.kv_heads, "source": "(user-provided, not fetched)"}
    elif args.model:
        name, cfg = args.model, MODEL_CONFIGS[args.model]
    else:
        ap.error("specify a model name, --custom, or --list")
        return

    head_dim, checks = check_model(name, cfg)
    ok = print_report(name, cfg, head_dim, checks)

    if args.diagnose:
        sys.path.insert(0, str(Path(__file__).parent))
        import amp_diagnose
        print("\n--- amp_diagnose.py static scan of kernels/flash_attn.cu(.cuh) ---")
        root = Path(__file__).parent.parent
        findings = (amp_diagnose.analyze_file(root / "kernels" / "flash_attn.cu") +
                    amp_diagnose.analyze_file(root / "kernels" / "flash_attn.cuh"))
        patterns = amp_diagnose.load_patterns()
        if findings:
            amp_diagnose.print_report(findings, patterns)
        else:
            print("No suspicious patterns found by static analysis. Note: flash_attn.cu uses "
                  "dynamic shared memory with pointer-arithmetic indexing, which this analyzer's "
                  "regex-based checks don't cover (see README) - a clean result here is expected "
                  "and is not the same as a positive correctness proof.")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
