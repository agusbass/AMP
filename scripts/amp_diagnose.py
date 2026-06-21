#!/usr/bin/env python3
"""amp_diagnose.py - static auto-diagnosis for AMP GPU kernels (futurework.md V2.1).

Given a numerical mismatch from amp_verify_matmul or parity_check.py, this
tool does NOT re-run anything on the GPU — it statically scans the kernel
source for known bug *patterns* (see scripts/bug_patterns.json) and points at
the specific lines worth checking by hand, the same way a linter would.

It cannot prove a kernel is buggy or automatically fix it (see the honesty
note in futurework.md: GPU numerical error is statistical, not logical, and
a wrong auto-fix can break correct code). It only narrows down where to look.

Coverage: the dimension-mismatch and missing-__syncthreads() checks rely on
statically-declared 2D shared arrays (`__shared__ T name[D1][D2]`) indexed
with `name[i][j]` — the idiom matmul.cu/.cuh use. Kernels that use dynamic
shared memory with pointer arithmetic (flash_attn.cu's `extern __shared__`
+ lambda accessors) are scanned without error but won't trigger those two
checks; only the tile-size check applies there.

Usage:
    python3 scripts/amp_diagnose.py kernels/matmul.cu kernels/matmul.cuh
    python3 scripts/amp_diagnose.py kernels/matmul.cu --tile 32,32,32
    python3 scripts/amp_diagnose.py kernels/matmul.cu --json
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

PATTERNS_PATH = Path(__file__).parent / "bug_patterns.json"

THREAD_INDEX_VARS = {"tx", "ty", "lane", "row", "col",
                      "threadIdx.x", "threadIdx.y", "threadIdx.z"}

SHARED_DECL_RE = re.compile(
    r"__shared__\s+[\w:&<>\*\s]+?\s+(\w+)\s*\[\s*(\w+)\s*\](?:\s*\[\s*(\w+)\s*\])?")
FOR_LOOP_RE = re.compile(
    r"for\s*\(\s*int\s+(\w+)\s*=\s*[^;]+;\s*\1\s*<\s*([\w]+)")
SYNC_RE = re.compile(r"__syncthreads\s*\(\s*\)|__syncwarp\s*\(\s*\)")
KERNEL_START_RE = re.compile(r"__global__")
ACCESS_RE_TMPL = r"\b{name}\s*\[\s*([^\]]+?)\s*\]\s*\[\s*([^\]]+?)\s*\]"
ASSIGN_RE = re.compile(r"=(?!=)")


@dataclass
class Finding:
    file: str
    line: int
    pattern_id: str
    code: str
    extra: str = ""


def load_patterns():
    db = json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))
    return {p["id"]: p for p in db["patterns"]}


def kernel_scopes(lines):
    """Split a source file into (start_line, end_line) ranges, one per
    __global__ kernel body, using __global__ markers as scope boundaries
    (good enough for AMP's flat kernel style; not a real C++ parser)."""
    starts = [i for i, l in enumerate(lines) if KERNEL_START_RE.search(l)]
    scopes = []
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        scopes.append((s, e))
    return scopes


def find_shared_decls(lines, lo, hi):
    decls = []
    for i in range(lo, hi):
        m = SHARED_DECL_RE.search(lines[i])
        if m:
            name, d1, d2 = m.group(1), m.group(2), m.group(3)
            decls.append({"name": name, "d1": d1, "d2": d2, "line": i})
    return decls


def find_for_loop_bounds(lines, lo, hi):
    """var -> loop bound symbol, for simple `for (int v = x; v < BOUND; ...)`."""
    bounds = {}
    for i in range(lo, hi):
        m = FOR_LOOP_RE.search(lines[i])
        if m:
            bounds[m.group(1)] = m.group(2)
    return bounds


def is_write(line, name):
    """True if `name[..][..]` is the assignment target on this line."""
    pat = re.compile(ACCESS_RE_TMPL.format(name=re.escape(name)))
    for m in pat.finditer(line):
        after = line[m.end():].lstrip()
        if after.startswith("=") and not after.startswith("=="):
            return True
    return False


def has_access(line, name):
    return re.search(ACCESS_RE_TMPL.format(name=re.escape(name)), line) is not None


def detect_dim_mismatch(lines, lo, hi, decl, loop_bounds, file, tile_values, findings):
    name, d1, d2 = decl["name"], decl["d1"], decl["d2"]
    if d2 is None:
        return
    pat = re.compile(ACCESS_RE_TMPL.format(name=re.escape(name)))
    for i in range(decl["line"] + 1, hi):
        for m in pat.finditer(lines[i]):
            idx1, idx2 = m.group(1).strip(), m.group(2).strip()
            for idx_expr, decl_dim in ((idx1, d1), (idx2, d2)):
                if idx_expr in THREAD_INDEX_VARS or idx_expr not in loop_bounds:
                    continue
                bound = loop_bounds[idx_expr]
                if bound == decl_dim:
                    continue
                # if both sides resolve to the same numeric value via --tile, no bug
                if tile_values and bound in tile_values and decl_dim in tile_values:
                    if tile_values[bound] == tile_values[decl_dim]:
                        continue
                findings.append(Finding(
                    file, i + 1, "shared_mem_dim_mismatch", lines[i].strip(),
                    f"loop var '{idx_expr}' bounded by '{bound}' indexes "
                    f"'{name}' declared as [{d1}][{d2}] (dim mismatch on '{decl_dim}')"))


def detect_sync_issues(lines, lo, hi, shared_names, file, findings):
    state = {n: "unknown" for n in shared_names}  # unknown|written|synced|read
    for i in range(lo, hi):
        line = lines[i]
        if SYNC_RE.search(line):
            for n in state:
                state[n] = "synced"
            continue
        for n in shared_names:
            if not has_access(line, n):
                continue
            if is_write(line, n):
                if state[n] == "read":
                    findings.append(Finding(
                        file, i + 1, "loop_carried_overwrite", line.strip(),
                        f"'{n}' written again before a __syncthreads() followed "
                        f"the previous read — other threads may still be reading it"))
                state[n] = "written"
            else:
                if state[n] == "written":
                    findings.append(Finding(
                        file, i + 1, "missing_syncthreads", line.strip(),
                        f"'{n}' read here without a __syncthreads() since it was "
                        f"last written"))
                    state[n] = "flagged"
                elif state[n] == "synced":
                    state[n] = "read"


def detect_bank_conflict(lines, lo, hi, decl, file, tile_values, findings):
    if not tile_values:
        return
    name, d1, d2 = decl["name"], decl["d1"], decl["d2"]
    if d2 is None:
        return
    d2_val = tile_values.get(d2)
    if d2_val is None or d2_val % 32 != 0:
        return
    pat = re.compile(ACCESS_RE_TMPL.format(name=re.escape(name)))
    for i in range(decl["line"] + 1, hi):
        for m in pat.finditer(lines[i]):
            idx1 = m.group(1).strip()
            if idx1 in THREAD_INDEX_VARS:
                findings.append(Finding(
                    file, i + 1, "bank_conflict_stride", lines[i].strip(),
                    f"'{name}' indexed [{idx1}][...] with trailing dim "
                    f"{d2}={d2_val} (multiple of 32 banks)"))


def detect_tile_too_large(lines, file, tile_values, findings):
    if not tile_values:
        return
    bm, bn = tile_values.get("BM"), tile_values.get("BN")
    if bm and bn and bm * bn > 1024:
        findings.append(Finding(
            file, 0, "tile_too_large_for_block",
            f"BM={bm} BN={bn} -> {bm*bn} threads/block",
            "exceeds the 1024 threads/block hardware limit"))


def strip_line_comment(line):
    """Drop a trailing // comment so its text can't fool the regexes below
    (e.g. a comment that mentions __syncthreads() by name)."""
    idx = line.find("//")
    return line[:idx] if idx != -1 else line


def analyze_file(path, tile_values=None):
    text = Path(path).read_text(encoding="utf-8")
    lines = [strip_line_comment(l) for l in text.splitlines()]
    findings = []
    scopes = kernel_scopes(lines) or [(0, len(lines))]
    for lo, hi in scopes:
        decls = find_shared_decls(lines, lo, hi)
        loop_bounds = find_for_loop_bounds(lines, lo, hi)
        shared_names = [d["name"] for d in decls]
        for d in decls:
            detect_dim_mismatch(lines, lo, hi, d, loop_bounds, str(path), tile_values, findings)
            detect_bank_conflict(lines, lo, hi, d, str(path), tile_values, findings)
        detect_sync_issues(lines, lo, hi, shared_names, str(path), findings)
    detect_tile_too_large(lines, str(path), tile_values, findings)
    return findings


def parse_tile_arg(s):
    if not s:
        return None
    parts = [int(x) for x in s.split(",")]
    names = ["BM", "BN", "BK"][:len(parts)]
    return dict(zip(names, parts))


def print_report(findings, patterns):
    if not findings:
        print("No suspicious patterns found by static analysis.")
        return 0
    print("-" * 66)
    worst = 0
    for f in findings:
        p = patterns.get(f.pattern_id, {})
        sev = p.get("severity", "info")
        worst = max(worst, {"info": 0, "high": 1}.get(sev, 0))
        print(f"  [{sev.upper()}] {f.file}:{f.line} - {p.get('name', f.pattern_id)}")
        print(f"     Suspicious code: {f.code}")
        if f.extra:
            print(f"     Possible cause: {f.extra}")
        if p.get("suggestion"):
            print(f"     Suggestion: {p['suggestion']}")
        print("-" * 66)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help="kernel source files (.cu/.cuh) to scan")
    ap.add_argument("--tile", help="BM,BN,BK values to substitute for bound checks")
    ap.add_argument("--json", action="store_true", help="emit findings as JSON")
    args = ap.parse_args()

    tile_values = parse_tile_arg(args.tile)
    patterns = load_patterns()

    all_findings = []
    for f in args.files:
        all_findings.extend(analyze_file(f, tile_values))

    if args.json:
        print(json.dumps([f.__dict__ for f in all_findings], indent=2))
        sys.exit(1 if any(patterns.get(f.pattern_id, {}).get("severity") == "high"
                           for f in all_findings) else 0)

    worst = print_report(all_findings, patterns)
    sys.exit(1 if worst else 0)


if __name__ == "__main__":
    main()
