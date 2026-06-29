#!/usr/bin/env python3
"""amp_suggest_fix.py - turn an amp_diagnose.py finding into a concrete
suggested fix.

For the two pattern types that have an unambiguous structural repair
(shared_mem_dim_mismatch, missing_syncthreads) this produces a deterministic,
mechanical fix - no LLM needed, because the "right" answer is implied by the
kernel's own declared shapes (the array's declared dimension symbol, or "a
__syncthreads() belongs right before this read"). For every other pattern
(bank_conflict_stride, register_spill_risk, tile_too_large_for_block) there is
no single mechanical fix, so this only emits the human-readable explanation
already attached to the pattern in bug_patterns.json - no fix is fabricated.

If ANTHROPIC_API_KEY is set, each non-mechanically-fixable finding's
explanation is additionally expanded in natural language via the Claude API.
This is optional and the tool degrades to the deterministic output above
when no key is present. A fix fabricated without a clear structural reason
is exactly the failure mode this project wants to avoid, so this tool never
invents a diff it can't justify from the kernel's own declared shapes.

Usage:
    python3 scripts/amp_suggest_fix.py kernels/matmul.cu kernels/matmul.cuh --tile 32,16,32
    python3 scripts/amp_suggest_fix.py kernels/matmul.cu --tile 32,16,32 --json > fix.json
"""
import argparse
import difflib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
import amp_diagnose  # noqa: E402

FIXABLE_PATTERNS = {"shared_mem_dim_mismatch", "missing_syncthreads"}
DIM_MISMATCH_EXTRA_RE = re.compile(r"bounded by '(\w+)'.*on '(\w+)'")
FOR_LOOP_BOUND_RE_TMPL = r"(\b\w+\s*=[^;]*;\s*\w+\s*<\s*){wrong}(\s*;)"


def group_dim_mismatch_fixes(findings):
    """Findings of this pattern on the same line, all agreeing on the same
    wrong/correct bound symbols, collapse into a single line-level fix.
    Disagreement on the same line means the fix isn't unambiguous, so it's
    skipped rather than guessed."""
    by_line = {}
    for f in findings:
        if f.pattern_id != "shared_mem_dim_mismatch":
            continue
        m = DIM_MISMATCH_EXTRA_RE.search(f.extra)
        if not m:
            continue
        wrong, correct = m.group(1), m.group(2)
        by_line.setdefault((f.file, f.line), set()).add((wrong, correct))
    fixes = []
    for (file, line), pairs in by_line.items():
        if len(pairs) == 1:
            wrong, correct = next(iter(pairs))
            fixes.append({"file": file, "line": line, "wrong": wrong, "correct": correct})
    return fixes


def apply_dim_mismatch_fix(lines, fix):
    i = fix["line"] - 1
    pat = re.compile(FOR_LOOP_BOUND_RE_TMPL.format(wrong=re.escape(fix["wrong"])))
    new_line, n = pat.subn(r"\g<1>" + fix["correct"] + r"\g<2>", lines[i])
    if n == 0:
        return None
    new_lines = list(lines)
    new_lines[i] = new_line
    return new_lines


def group_missing_sync_fixes(findings):
    return sorted({f.line for f in findings if f.pattern_id == "missing_syncthreads"})


def apply_missing_sync_fix(lines, line_no):
    i = line_no - 1
    indent = re.match(r"\s*", lines[i]).group(0)
    new_lines = list(lines)
    new_lines.insert(i, f"{indent}__syncthreads();")
    return new_lines


def explain_with_llm(pattern, finding):
    """Optional natural-language expansion via the Claude API. Returns None
    (silently) if no API key is configured - the tool must work fully
    offline without this. Never raises: a flaky network call must not break
    the deterministic part of this tool."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    prompt = (
        f"A static GPU-kernel analyzer flagged this:\n"
        f"Pattern: {pattern.get('name')}\n"
        f"Code: {finding.code}\n"
        f"Cause: {finding.extra}\n"
        f"In 2 sentences, explain to a developer porting CUDA to HIP why this "
        f"matters numerically and what to check.")
    body = json.dumps({
        "model": "claude-3-5-haiku-latest",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body, method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return data["content"][0]["text"]
    except Exception as e:
        return f"(LLM explanation unavailable: {e})"


def suggest(files, tile_values):
    patterns = amp_diagnose.load_patterns()
    results = []

    for path in files:
        findings = amp_diagnose.analyze_file(path, tile_values)
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        new_lines = list(lines)
        applied = []

        for fix in group_dim_mismatch_fixes(findings):
            patched = apply_dim_mismatch_fix(new_lines, fix)
            if patched:
                new_lines = patched
                applied.append(f"line {fix['line']}: '{fix['wrong']}' -> '{fix['correct']}'")

        # Insert from the bottom up so earlier line numbers stay valid as
        # __syncthreads() lines get inserted above them.
        for line_no in sorted(group_missing_sync_fixes(findings), reverse=True):
            new_lines = apply_missing_sync_fix(new_lines, line_no)
            applied.append(f"line {line_no}: inserted __syncthreads()")

        non_fixable = [f for f in findings if f.pattern_id not in FIXABLE_PATTERNS]
        explanations = []
        for f in non_fixable:
            p = patterns.get(f.pattern_id, {})
            explanations.append({
                "line": f.line, "pattern_id": f.pattern_id,
                "description": p.get("description", ""),
                "suggestion": p.get("suggestion", ""),
                "llm_explanation": explain_with_llm(p, f) if p else None,
            })

        if applied:
            new_text = "\n".join(new_lines) + "\n"
            diff = "".join(difflib.unified_diff(
                [l + "\n" for l in lines], [l + "\n" for l in new_lines],
                fromfile=path, tofile=path + " (suggested fix)"))
        else:
            new_text, diff = None, ""

        results.append({
            "file": path,
            "applied_fixes": applied,
            "diff": diff,
            "new_text": new_text,
            "unfixable_findings": explanations,
        })
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tile", help="BM,BN,BK")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    tile_values = amp_diagnose.parse_tile_arg(args.tile)
    results = suggest(args.files, tile_values)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    any_fix = False
    for r in results:
        if r["applied_fixes"]:
            any_fix = True
            print(f"=== {r['file']} - suggested fix ===")
            for a in r["applied_fixes"]:
                print(f"  - {a}")
            print(r["diff"])
        for u in r["unfixable_findings"]:
            print(f"=== {r['file']}:{u['line']} - no mechanical fix ({u['pattern_id']}) ===")
            print(f"  {u['description']}")
            print(f"  Suggestion: {u['suggestion']}")
            if u["llm_explanation"]:
                print(f"  LLM: {u['llm_explanation']}")
    if not any_fix:
        print("No mechanically-fixable findings (set ANTHROPIC_API_KEY for "
              "LLM explanations of any non-fixable findings above, if any).")


if __name__ == "__main__":
    main()
