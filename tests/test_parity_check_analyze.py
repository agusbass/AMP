"""Tests that scripts/parity_check.py's --analyze path actually fires on a
genuine cross-vendor FAIL -- not just that the diff math is correct.

Every real parity_check.py run in this project's validation log has PASSed
(see docs/VALIDATION.md), so the auto-diagnose-on-FAIL branch was never
exercised by real hardware output. These fixtures force a FAIL
deterministically, with no GPU needed, to close that gap.
"""
import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "parity_check.py"
FIXTURE_A = ROOT / "tests" / "fixtures" / "synthetic_fail_a.json"
FIXTURE_B = ROOT / "tests" / "fixtures" / "synthetic_fail_b.json"


def run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(FIXTURE_A), str(FIXTURE_B), *args],
        capture_output=True, text=True,
    )


class TestAnalyzeFiresOnRealFail:
    def test_mismatched_outputs_are_reported_as_fail(self):
        r = run()
        assert r.returncode == 1
        assert "FAIL" in r.stdout
        assert "FAILED CROSS-VENDOR PARITY" in r.stdout

    def test_analyze_flag_runs_static_diagnosis_against_real_kernel(self):
        r = run("--analyze")
        assert r.returncode == 1
        assert "Auto-diagnosis" in r.stdout
        # Scans the real shipped kernel (default --sources), not a fixture
        assert "kernels/matmul.cuh" in r.stdout

    def test_without_analyze_flag_no_diagnosis_section_printed(self):
        r = run()
        assert "Auto-diagnosis" not in r.stdout
