#!/usr/bin/env python3
"""amp_fix.py - V3: apply a suggested fix from amp_suggest_fix.py, with a
human in the loop.

This deliberately requires an explicit --yes to write anything: an
auto-fix applied without review can break a kernel that was already
correct, so a human stays in the loop. Without --yes it only shows the
diff (dry run, the default). A .bak backup of every changed file is
written before any change.

This tool does NOT rebuild or re-run the GPU correctness tests after
patching - it only applied a structural source change. Rebuilding and
re-running amp_verify_matmul / parity_check.py is still required to confirm
the fix is actually correct on real hardware.

Usage:
    python3 scripts/amp_suggest_fix.py kernels/matmul.cu --tile 32,16,32 --json > fix.json
    python3 scripts/amp_fix.py fix.json          # dry run, shows diff(s)
    python3 scripts/amp_fix.py fix.json --yes    # actually writes the fix(es)
"""
import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fix_json", help="output of amp_suggest_fix.py --json")
    ap.add_argument("--yes", action="store_true",
                     help="apply the fix(es) (default: dry run only)")
    args = ap.parse_args()

    results = json.loads(Path(args.fix_json).read_text(encoding="utf-8"))
    applied_any = False

    for r in results:
        if not r.get("applied_fixes"):
            continue
        print(f"=== {r['file']} ===")
        for a in r["applied_fixes"]:
            print(f"  - {a}")
        print(r["diff"])

        if not args.yes:
            print(f"  (dry run - rerun with --yes to write {r['file']})")
            continue

        path = Path(r["file"])
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(r["new_text"], encoding="utf-8")
        print(f"  Applied. Original backed up to {backup}")
        applied_any = True

    if applied_any:
        print("\nNow rebuild and re-run amp_verify_matmul / parity_check.py to "
              "confirm the fix is actually correct - this tool only applied a "
              "structural patch, it did not re-verify on a GPU.")
    sys.exit(0)


if __name__ == "__main__":
    main()
