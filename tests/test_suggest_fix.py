#!/usr/bin/env python3
"""Tests for scripts/amp_suggest_fix.py (V2.2) and scripts/amp_fix.py (V3).

Locks in the end-to-end loop, entirely offline (no GPU, no ANTHROPIC_API_KEY
required): diagnose the buggy fixture -> suggest a mechanical fix -> apply it
with --yes -> re-diagnose and confirm the two HIGH findings are gone.

Run: python3 tests/test_suggest_fix.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import amp_diagnose  # noqa: E402
import amp_suggest_fix  # noqa: E402

ROOT = Path(__file__).parent.parent
BUGGY_FIXTURE = ROOT / "tests" / "fixtures" / "buggy_matmul_fixture.cuh"
TILE = {"BM": 32, "BN": 16, "BK": 32}


class TestSuggestFix(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.scratch = Path(self.tmpdir) / "buggy.cuh"
        shutil.copy(BUGGY_FIXTURE, self.scratch)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_suggest_produces_both_mechanical_fixes(self):
        results = amp_suggest_fix.suggest([str(self.scratch)], TILE)
        self.assertEqual(len(results), 1)
        applied = results[0]["applied_fixes"]
        self.assertTrue(any("BN" in a and "BK" in a for a in applied))
        self.assertTrue(any("__syncthreads" in a for a in applied))

    def test_no_llm_call_without_api_key(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        results = amp_suggest_fix.suggest([str(self.scratch)], TILE)
        for u in results[0]["unfixable_findings"]:
            self.assertIsNone(u["llm_explanation"])

    def test_apply_with_yes_clears_high_findings(self):
        results = amp_suggest_fix.suggest([str(self.scratch)], TILE)
        fix_json = Path(self.tmpdir) / "fix.json"
        fix_json.write_text(json.dumps(results), encoding="utf-8")

        # dry run must not modify the file
        before = self.scratch.read_text(encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "amp_fix.py"), str(fix_json)],
                        check=True, capture_output=True)
        self.assertEqual(self.scratch.read_text(encoding="utf-8"), before)

        subprocess.run([sys.executable, str(ROOT / "scripts" / "amp_fix.py"),
                         str(fix_json), "--yes"], check=True, capture_output=True)
        self.assertTrue(self.scratch.with_suffix(".cuh.bak").exists())

        findings = amp_diagnose.analyze_file(self.scratch, TILE)
        patterns = amp_diagnose.load_patterns()
        high = [f for f in findings if patterns[f.pattern_id]["severity"] == "high"]
        self.assertEqual(high, [])


if __name__ == "__main__":
    unittest.main()
