#!/usr/bin/env python3
"""Cross-vendor GEMM parity check.

Usage:
    python3 scripts/parity_check.py cuda_dump.json hip_dump.json

Each dump is produced by `amp_parity_dump` (tests/parity_dump.cpp), run once
on NVIDIA hardware (CUDA build) and once on AMD hardware (HIP build), with
identical shapes/seed. This script does NOT re-derive correctness from a CPU
reference — both dumps already carry max_rel_err_vs_cpu for that. Its job is
the comparison neither vendor's own correctness check can do on its own:
diffing the two vendors' actual outputs against each other, and reporting the
GFLOPS gap per kernel tile config.
"""
import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

REL_ERR_THRESHOLD = 1e-3

# Default kernel sources to statically scan when a tile config fails parity
# (see amp_diagnose.py / futurework.md V2.1 — auto-diagnosis, not auto-fix).
DEFAULT_DIAGNOSE_SOURCES = ["kernels/matmul.cu", "kernels/matmul.cuh"]


def load(path):
    return json.loads(Path(path).read_text())


def tile_key(t):
    return (t["bm"], t["bn"], t["bk"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump_a")
    ap.add_argument("dump_b")
    ap.add_argument("--analyze", action="store_true",
                     help="on any FAIL, run static auto-diagnosis "
                          "(amp_diagnose.py) against the kernel source for "
                          "that tile config")
    ap.add_argument("--sources", nargs="+", default=DEFAULT_DIAGNOSE_SOURCES,
                     help="kernel source files to scan with --analyze")
    args = ap.parse_args()

    a = load(args.dump_a)
    b = load(args.dump_b)

    if (a["M"], a["N"], a["K"], a["seed"]) != (b["M"], b["N"], b["K"], b["seed"]):
        print("ERROR: dumps use different shapes/seed — not comparable.")
        print(f"  A: M={a['M']} N={a['N']} K={a['K']} seed={a['seed']}")
        print(f"  B: M={b['M']} N={b['N']} K={b['K']} seed={b['seed']}")
        sys.exit(1)

    tiles_b = {tile_key(t): t for t in b["tiles"]}

    print(f"\nCross-vendor parity: {a['vendor']} (A) vs {b['vendor']} (B)")
    print(f"Shape: M={a['M']} N={a['N']} K={a['K']}  seed={a['seed']}\n")
    print(f"{'tile':<14} {'A GFLOPS':>10} {'B GFLOPS':>10} {'B/A ratio':>10} "
          f"{'max_abs_err':>12} {'max_rel_err':>12}  status")
    print("-" * 86)

    failures = 0
    failing_tiles = []
    for ta in a["tiles"]:
        key = tile_key(ta)
        tb = tiles_b.get(key)
        if tb is None:
            print(f"{key!s:<14} -- B has no matching tile config, skipped")
            continue

        out_a, out_b = ta["output"], tb["output"]
        if len(out_a) != len(out_b):
            print(f"{key!s:<14} -- output length mismatch, skipped")
            failures += 1
            continue

        max_abs_err = max(abs(x - y) for x, y in zip(out_a, out_b))
        max_rel_err = max(
            abs(x - y) / max(1.0, abs(y)) for x, y in zip(out_a, out_b)
        )
        ratio = tb["gflops"] / ta["gflops"] if ta["gflops"] else float("inf")
        ok = max_rel_err < REL_ERR_THRESHOLD
        if not ok:
            failures += 1
            failing_tiles.append(key)

        tile_str = f"({ta['bm']},{ta['bn']},{ta['bk']})"
        print(f"{tile_str:<14} {ta['gflops']:>10.1f} {tb['gflops']:>10.1f} {ratio:>9.2f}x "
              f"{max_abs_err:>12.6f} {max_rel_err:>12.6f}  {'PASS' if ok else 'FAIL'}")

    print()
    if failures == 0:
        print(f"ALL TILE CONFIGS CROSS-VENDOR PARITY OK (rel_err < {REL_ERR_THRESHOLD})")
    else:
        print(f"{failures} TILE CONFIG(S) FAILED CROSS-VENDOR PARITY — "
              f"{a['vendor']} and {b['vendor']} disagree beyond tolerance")

    if args.analyze and failing_tiles:
        sys.path.insert(0, str(Path(__file__).parent))
        import amp_diagnose

        patterns = amp_diagnose.load_patterns()
        print("\n--- Auto-diagnosis (static analysis, not a proof) ---")
        for bm, bn, bk in failing_tiles:
            print(f"\nTile ({bm},{bn},{bk}):")
            tile_values = {"BM": bm, "BN": bn, "BK": bk}
            findings = []
            for src in args.sources:
                findings.extend(amp_diagnose.analyze_file(src, tile_values))
            amp_diagnose.print_report(findings, patterns)

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
