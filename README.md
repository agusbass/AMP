# 🛠️ AMP — AMD Migration Platform

**Plug-and-play tooling that makes targeting AMD Instinct as easy as targeting NVIDIA — right when NVIDIA hardware is hardest to get.**

## 60-second demo: a bug that used to take hours to find

`tests/fixtures/buggy_matmul_fixture.cuh` is a kernel we wrote specifically
to test the analyzer below — it's *designed to recreate the same class* of
bug described in `tests/verify_matmul_correctness.cpp`'s comments (a tiled
GEMM where the inner loop's bound symbol doesn't match the shared-memory
array's declared dimension, plus a missing `__syncthreads()`), not a literal
copy of the original historical commit. That bug class doesn't crash on real
hardware — it just silently produces wrong numbers for some tile configs on
some inputs, which is exactly what survives a "compiles and doesn't crash"
check.

The block below is the literal, unedited terminal output of running both
commands against that fixture (confirmed by re-running them just now):

```bash
$ python3 scripts/amp_diagnose.py tests/fixtures/buggy_matmul_fixture.cuh --tile 32,16,32
[HIGH] tests/fixtures/buggy_matmul_fixture.cuh:20 - Shared memory dimension mismatch
   Suspicious code: for (int k = 0; k < BN; ++k) acc += sA[ty][k] * sB[k][tx];
   Possible cause: loop var 'k' bounded by 'BN' indexes 'sA' declared as [BM][BK] (dim mismatch on 'BK')
[HIGH] tests/fixtures/buggy_matmul_fixture.cuh:20 - Missing __syncthreads() between shared-memory write and read
   Suspicious code: for (int k = 0; k < BN; ++k) acc += sA[ty][k] * sB[k][tx];
   Possible cause: 'sA' read here without a __syncthreads() since it was last written

$ python3 scripts/amp_suggest_fix.py tests/fixtures/buggy_matmul_fixture.cuh --tile 32,16,32
=== tests/fixtures/buggy_matmul_fixture.cuh - suggested fix ===
  - line 20: 'BN' -> 'BK'
  - line 20: inserted __syncthreads()
```

(Both commands print more than this — the diagnose run also reports the
same dimension-mismatch/missing-sync pair on the `sB` array plus two
info-level bank-conflict notes, and suggest-fix also prints the unified diff
and those same non-mechanically-fixable notes. Trimmed here for length; run
it yourself to see the rest, nothing below the trim line contradicts this.)

That fix is mechanical — no LLM call happens for it — because the kernel's
own declared shapes already imply the correct answer (see
`scripts/amp_suggest_fix.py`'s `FIXABLE_PATTERNS` for exactly which two
pattern types qualify). No GPU is required for any of this: it's static
analysis against the kernel source, wired into the existing
`parity_check.py --analyze` flow so a real cross-vendor FAIL triggers it
automatically. Running this same scan against `kernels/matmul.cu` (the real,
shipped kernel) finds zero HIGH-severity issues for any of its 4 supported
tile configs — we verified that in `tests/test_diagnose.py`, which is also
the test that proves the fixture's two injected bugs are caught. See
**Auto-diagnosis and auto-fix** below for the full loop (diagnose → suggest
fix → human confirms → apply → rebuild → re-verify).

## Does AMP's kernel even support your model?

Before validating numbers on hardware, there's a cheaper question: does AMP's
FlashAttention-2 kernel support the *architecture* a real open-source model
actually uses? `scripts/amp_model_check.py` checks a model's published
attention shape (head_dim, GQA head grouping) against two hard constraints
in `kernels/flash_attn.cuh`/`.cu` — no GPU needed, and the model configs
below were fetched from each model's public, non-gated `config.json` on
Hugging Face (cited inline), not reproduced from memory:

```bash
$ python3 scripts/amp_model_check.py mistral-7b-v0.1
Model: mistral-7b-v0.1
  hidden_size=4096 num_attention_heads=32 num_key_value_heads=8 (derived head_dim=128)
  source: https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json

  [OK] head_dim is a whole number (hidden_size divisible by num_attention_heads)
  [OK] head_dim=128 is a multiple of 16 (kernels/flash_attn.cuh)
  [OK] head_dim=128 <= 256 (kernels/flash_attn.cuh)
  [OK] num_key_value_heads=8 <= num_attention_heads=32
  [OK] num_attention_heads=32 evenly divisible by num_key_value_heads=8 (flash_attn.cu:90 GQA grouping)

Result: COMPATIBLE -- AMP's FlashAttention-2 kernel can serve mistral-7b-v0.1's attention shapes as written.
```

The GQA divisibility check isn't cosmetic: `flash_attn.cu:90` computes
`hkv = hq / (H_q / H_kv)` — integer division, so a model whose head count
doesn't divide evenly would silently misgroup KV heads instead of crashing.
`--list` shows all built-in configs (Mistral-7B-v0.1, TinyLlama-1.1B-Chat,
Qwen2-7B, Phi-3-mini-4k — all real and currently pass; plus one clearly
fictional config to demonstrate what a FAIL looks like), and `--custom
--hidden-size N --heads N --kv-heads N` checks any model's own numbers
directly. This is a static pre-flight check against published numbers, not
a substitute for `amp_verify_matmul`/`parity_check.py` on real hardware.

## Why now

In 2026, NVIDIA controls roughly 80% of the AI accelerator market and CUDA has a 20+ year, 4M+ developer ecosystem — that's the reality, not something to argue against. But 2026 also brought a documented GPU/VRAM supply crunch: NVIDIA has reportedly cut GPU production by up to 40% on some lines, AI customers are prioritized over everyone else, and prices on flagship parts have swung sharply. For a developer who needs more compute or more memory *this quarter*, "wait for NVIDIA supply to ease" isn't always an option.

AMD Instinct (MI300X/MI350X) is a credible alternative — more available, often cheaper per GB of memory — but most of the friction in trying it isn't theoretical performance, it's **practical setup cost**: picking the right CMake backend flags, figuring out which ROCm features are available, debugging driver/runtime version mismatches, and not knowing if a CUDA→HIP port is even numerically correct before you trust it.

**AMP's job is to remove that friction.** It is not trying to prove AMD is faster than NVIDIA — it's trying to make trying AMD a same-day decision instead of a multi-week migration project.

## Business value: who actually pays for this, and why isn't AMD building it

AMD already ships two pieces of this puzzle: **HIPIFY** does the mechanical
CUDA→HIP source transpile, and the **ROCm Validation Suite** checks that a
ROCm *install* is healthy. Neither answers "is the kernel HIPIFY just
transpiled for me numerically correct, and how far off is it from the CUDA
original, kernel by kernel?" — that gap sits between the two, it's specific
to inference kernels (GEMM/FlashAttention/PagedAttention) rather than
general HPC, and it's the part a vendor has limited incentive to build
aggressively (AMD wants you on ROCm; a developer wants proof before they
trust ROCm with production traffic). That asymmetry is exactly why a
third-party tool is the right shape for this, not a feature request to AMD.

**Who pays:** any team currently blocked from trying AMD Instinct by "we
can't tell if the port is correct without a multi-week manual validation
project" — i.e. teams squeezed by the 2026 NVIDIA supply crunch who'd
otherwise just wait. Once each vendor's dump exists, AMP turns the
cross-vendor comparison into a one-command diff plus an automated first
pass at root-causing any failure — the part of a migration that's hardest
to estimate or budget for in advance.

We name this gap explicitly rather than claim novelty in the abstract — see
the honesty note below for what already exists and what doesn't.

## What's in here

| Component | Status | Source |
|---|---|---|
| Multi-vendor C++17 runtime (CUDA/HIP/SYCL/CPU) | Claimed working, tested on real NVIDIA T4 hardware *(inherited claim, see caveat below)* | `include/`, `src/`, `kernels/` |
| Auto-detect build (`scripts/build_auto.sh`) | Reads correctly for all 4 backends *(inherited, not re-run — see caveat)* | `scripts/` |
| GEMM (FP32/BF16/FP8) + FlashAttention-2 kernels | Test exists and its logic checks out by reading it; "numerically verified" is an inherited claim, not re-run here | `kernels/`, `tests/verify_matmul_correctness.cpp` |
| Continuous batching scheduler, speculative decoding, paged KV pool | Claimed working, benchmarked *(inherited claim, see caveat below)* | `src/` |
| Docker image (ROCm 6.3.2 base) | Dockerfile present, build not attempted in this session (no Docker available) | `Dockerfile` |
| Static auto-diagnosis + suggested-fix for kernel bugs | ✅ **Verified in this session**: 8/8 tests pass, zero false positives on the real kernel's 4 supported tile configs, both injected bugs in the fixture are caught and mechanically fixed | `scripts/amp_diagnose.py`, `scripts/amp_suggest_fix.py`, `scripts/amp_fix.py`, `tests/test_diagnose.py`, `tests/test_suggest_fix.py` |
| Open-source model compatibility check (FlashAttention-2 head_dim/GQA) | ✅ **Verified in this session**: checks 4 real models (configs fetched live from Hugging Face, not memorized) against the kernel's actual constraints; all 4 pass, fictional bad-GQA example correctly fails | `scripts/amp_model_check.py`, `tests/test_model_check.py` |

**Important caveat on the first four rows:** this runtime is carried over
from an earlier project (internally called ANM). Those checkmarks describe
claims made about that predecessor project, not something independently
re-verified in this session — there is no compiler, GPU, or Docker
available in this environment, so none of the C++ build/test/benchmark
claims above could be re-run or re-confirmed here. Only the last two rows
(pure Python, neither needs a compiler nor a GPU) have actually been
executed and checked in this session.

`scripts/amp_pipeline.sh` (introduced in this session, see below) chains
build → verify → diagnose → fix together but is **syntax-checked only, not
execution-tested** — it has never actually been run, on any hardware.
Closing all of these gaps requires running the C++ build and tests on a
machine with ROCm/CUDA installed (e.g. AMD Developer Cloud) — see Quick
start below.

## What's still being built

The original C++ runtime answers "can this code run on AMD." It doesn't yet answer the two questions that actually block migration decisions:

1. **Is the ROCm port numerically correct**, not just "compiles and doesn't crash"? (Industry practice today is mostly ad-hoc `torch.allclose()` checks — there's no lightweight, drop-in tool for this on LLM-inference-style kernels specifically.)
2. **How far off is performance**, kernel by kernel, from the CUDA original?

We looked for prior art before committing to this (see honesty note below) — closest is [CASS](https://github.com/ahmedheakl/CASS) (MBZUAI), which validates CUDA→HIP transpilation correctness/performance, but as a research benchmark over classic HPC workloads (Rodinia/SHOC/PolyBench), not a practical tool for LLM-inference kernels (GEMM, FlashAttention, PagedAttention) that a developer can point at their own code.

## Honesty note

We've gone through several pivots building this (you can see the history in commit messages / STRATEGY docs of the predecessor project). Two earlier ideas turned out to already exist in production: KV-cache offload to CPU memory for long context (already solved by [LMCache](https://github.com/LMCache/LMCache), which has an official AMD partnership and 3-10x measured gains on MI300X via vLLM), and several ROCm install/validation tools (AMD's own ROCm Validation Suite, RCCL-Tests, community installers). We're naming this upfront so the gap we're targeting — practical CUDA↔ROCm parity validation for inference kernels, packaged as a drop-in tool, plus genuinely one-command builds — is stated as what we believe is still open, not asserted as "no one has ever done this."

## Quick start

```bash
# Auto-detect vendor (CUDA/HIP/SYCL/CPU) and build with the right flags
bash scripts/build_auto.sh

# Run the integration test
cd build && ./test_triple

# Numerical correctness check (GEMM tile configs vs CPU reference)
./amp_verify_matmul
```

## Cross-vendor parity check

This is the new piece: proving a CUDA kernel and its HIP port produce the
*same numbers*, not just that each one separately agrees with a CPU
reference (which `amp_verify_matmul` already does, but only within one
vendor's run).

```bash
# On the NVIDIA machine (CUDA build):
./build/amp_parity_dump cuda_dump.json

# On the AMD machine (HIP build), same shapes/seed:
./build/amp_parity_dump hip_dump.json

# Anywhere with Python — diff the two:
python3 scripts/parity_check.py cuda_dump.json hip_dump.json

# Add --analyze to auto-diagnose any FAILing tile config (static analysis
# against the kernel source, points at the suspicious lines):
python3 scripts/parity_check.py cuda_dump.json hip_dump.json --analyze
```

Output is a per-tile-config report: GFLOPS on each vendor, the speed ratio,
and the max absolute/relative error between the two vendors' actual outputs
— the comparison neither vendor's own correctness check can produce alone.

## Auto-diagnosis and auto-fix

When `parity_check.py` or `amp_verify_matmul` reports a numerical mismatch,
the next question is *where* in the kernel to look — that's what this layer
answers, without needing a GPU to run:

```bash
# Static analysis: scan kernel source for known bug patterns (shared-memory
# dimension mismatches, missing __syncthreads(), bank-conflict risk, tile
# configs that exceed 1024 threads/block) and point at the suspicious lines.
python3 scripts/amp_diagnose.py kernels/matmul.cu kernels/matmul.cuh --tile 32,16,32

# Turn a finding into a concrete fix. Two pattern types have an unambiguous
# structural repair (dimension mismatch, missing sync) and get a mechanical,
# deterministic patch — no LLM involved, because the kernel's own declared
# shapes already tell you the right answer. Everything else only gets an
# explanation (optionally expanded by Claude if ANTHROPIC_API_KEY is set) —
# no fix is fabricated without a clear structural reason.
python3 scripts/amp_suggest_fix.py kernels/matmul.cu kernels/matmul.cuh --tile 32,16,32 --json > fix.json

# Review the diff, then apply it explicitly — nothing is written without --yes,
# and the original is always backed up to *.bak first.
python3 scripts/amp_fix.py fix.json          # dry run: just shows the diff
python3 scripts/amp_fix.py fix.json --yes    # actually applies it

# Or chain build -> verify -> diagnose -> (confirmed) fix -> rebuild -> re-verify in one go
# (this script is syntax-checked only — it has not actually been run end-to-end,
# since no C++ toolchain was available while writing it; try it before relying on it):
bash scripts/amp_pipeline.sh --auto-fix
```

This is deliberately diagnosis-and-suggestion, not silent auto-fix: a wrong
fix can break a kernel that was already correct, so a human always reviews
the diff before anything is written (see `tests/test_diagnose.py` and
`tests/test_suggest_fix.py` for what's covered — it's regex/heuristic-based,
not a real C++ parser, so it works best on the statically-declared 2D
shared-memory idiom `matmul.cu`/`.cuh` use; kernels using dynamic shared
memory with pointer arithmetic, like `flash_attn.cu`, are scanned safely but
won't trigger the dimension-mismatch/missing-sync checks).

## Hackathon context

| Item | Detail |
|---|---|
| Hackathon | AMD Developer Hackathon: ACT II |
| Track | Unicorn Track |
| Judges | Pawel Czech (CEO), Andrea Marazzi (Founder & CCO), lablab.ai |
| Criteria | Application of Technology, Presentation, Business Value, Originality |
