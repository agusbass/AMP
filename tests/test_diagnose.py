#!/usr/bin/env python3
"""Tests for scripts/amp_diagnose.py. Pure static analysis, no GPU/compiler
needed. Locks in two things:
  1. The real (already-fixed) matmul kernel produces no HIGH severity finding
     for any of its 4 supported tile configs (no false positives).
  2. A deliberately broken fixture, recreating a known shared-memory
     dimension-mismatch plus missing-syncthreads bug class, is actually
     caught (no false negatives).

Run: python3 tests/test_diagnose.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import amp_diagnose  # noqa: E402

ROOT = Path(__file__).parent.parent
REAL_SOURCES = [ROOT / "kernels" / "matmul.cu", ROOT / "kernels" / "matmul.cuh"]
BUGGY_FIXTURE = ROOT / "tests" / "fixtures" / "buggy_matmul_fixture.cuh"


class TestRealKernelHasNoFalsePositives(unittest.TestCase):
    def test_all_supported_tiles_clean(self):
        for bm, bn, bk in [(16, 16, 16), (32, 16, 16), (32, 32, 16), (32, 32, 32)]:
            tile_values = {"BM": bm, "BN": bn, "BK": bk}
            findings = []
            for src in REAL_SOURCES:
                findings.extend(amp_diagnose.analyze_file(src, tile_values))
            high = [f for f in findings if
                    amp_diagnose.load_patterns()[f.pattern_id]["severity"] == "high"]
            self.assertEqual(high, [], f"unexpected HIGH finding for tile {tile_values}")


class TestBuggyFixtureIsCaught(unittest.TestCase):
    def setUp(self):
        # BN != BK is required to exercise the dimension-mismatch detector.
        self.findings = amp_diagnose.analyze_file(BUGGY_FIXTURE, {"BM": 32, "BN": 16, "BK": 32})
        self.ids = {f.pattern_id for f in self.findings}

    def test_dimension_mismatch_detected(self):
        self.assertIn("shared_mem_dim_mismatch", self.ids)

    def test_missing_syncthreads_detected(self):
        self.assertIn("missing_syncthreads", self.ids)

    def test_findings_point_at_the_actual_buggy_line(self):
        target_line = 20  # the dot-product loop with the BN/BK mix-up
        self.assertTrue(any(f.line == target_line for f in self.findings))


class TestCommentsDontFoolTheAnalyzer(unittest.TestCase):
    def test_comment_mentioning_syncthreads_is_stripped(self):
        line = "    sA[ty][tx] = x; // calls __syncthreads() conceptually"
        self.assertEqual(amp_diagnose.strip_line_comment(line),
                          "    sA[ty][tx] = x; ")


if __name__ == "__main__":
    unittest.main()
