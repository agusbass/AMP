#!/bin/bash
# amp_pipeline.sh - chains auto-build, auto-test, auto-diagnose, and
# optional auto-fix into one command.
#
# This only orchestrates pieces AMP already has locally:
#   build_auto.sh -> amp_verify_matmul -> amp_diagnose.py -> amp_suggest_fix.py
#   -> amp_fix.py -> rebuild -> re-verify
# It does NOT do the HIPIFY transpile step (that is a separate AMD-provided
# tool, out of scope for this repo) and it does NOT apply any source change
# without a human confirming it first, even with --auto-fix, since a wrong
# auto-fix can break a kernel that was already correct.
#
# Usage:
#   bash scripts/amp_pipeline.sh                  # build + verify only
#   bash scripts/amp_pipeline.sh --auto-fix        # also offer mechanical fixes, asks before applying
#   bash scripts/amp_pipeline.sh --auto-fix --yes  # apply mechanical fixes without asking (CI use only)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MAX_FIX_ROUNDS=3

AUTO_FIX=0
ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --auto-fix) AUTO_FIX=1 ;;
        --yes) ASSUME_YES=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

run_build() {
    echo "=== [1/3] Build ==="
    bash "$SCRIPT_DIR/build_auto.sh"
}

run_verify() {
    echo ""
    echo "=== [2/3] Numerical correctness (amp_verify_matmul) ==="
    if [ -x "$PROJECT_DIR/build/amp_verify_matmul" ]; then
        "$PROJECT_DIR/build/amp_verify_matmul"
        return $?
    else
        echo "amp_verify_matmul not built for this backend (CPU fallback has no GPU kernels) - skipping."
        return 0
    fi
}

run_diagnose_and_maybe_fix() {
    echo ""
    echo "=== [3/3] Auto-diagnosis ==="
    python3 "$SCRIPT_DIR/amp_diagnose.py" "$PROJECT_DIR/kernels/matmul.cu" "$PROJECT_DIR/kernels/matmul.cuh"
    local diag_status=$?

    if [ "$AUTO_FIX" -ne 1 ]; then
        echo ""
        echo "Re-run with --auto-fix to let amp_suggest_fix.py propose a mechanical patch for the findings above."
        return $diag_status
    fi

    local fix_json="$PROJECT_DIR/build/amp_fix.json"
    python3 "$SCRIPT_DIR/amp_suggest_fix.py" "$PROJECT_DIR/kernels/matmul.cu" "$PROJECT_DIR/kernels/matmul.cuh" --json > "$fix_json"
    python3 "$SCRIPT_DIR/amp_fix.py" "$fix_json"   # dry run: always show the diff first

    if [ "$ASSUME_YES" -ne 1 ]; then
        read -r -p "Apply the fix(es) shown above? [y/N] " reply
        case "$reply" in
            [yY]*) ;;
            *) echo "Not applying. Fix this by hand, or re-run with --yes for unattended CI use."; return $diag_status ;;
        esac
    fi

    python3 "$SCRIPT_DIR/amp_fix.py" "$fix_json" --yes
    return 0
}

run_build
if ! run_verify; then
    if run_diagnose_and_maybe_fix; then
        if [ "$AUTO_FIX" -eq 1 ]; then
            echo ""
            echo "=== Rebuilding after applied fix ==="
            run_build
            run_verify
            exit $?
        fi
    fi
    exit 1
fi

echo ""
echo "AMP pipeline: build + verify passed."
