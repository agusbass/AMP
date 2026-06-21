#!/usr/bin/env python3
"""Tests for scripts/amp_model_check.py.

Locks in: real fetched model configs are all reported compatible (head_dim
multiple-of-16/<=256 and GQA divisibility, per kernels/flash_attn.cuh and
flash_attn.cu:90), and the fictional bad-GQA example actually fails.

Run: python3 tests/test_model_check.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import amp_model_check  # noqa: E402


class TestRealModelsAreCompatible(unittest.TestCase):
    def test_all_builtin_real_models_pass(self):
        for name, cfg in amp_model_check.MODEL_CONFIGS.items():
            if name == "example-bad-gqa-fictional":
                continue
            _, checks = amp_model_check.check_model(name, cfg)
            failing = [desc for desc, ok, _ in checks if not ok]
            self.assertEqual(failing, [], f"{name} unexpectedly failed: {failing}")


class TestFictionalBadGqaFails(unittest.TestCase):
    def test_uneven_gqa_is_flagged(self):
        cfg = amp_model_check.MODEL_CONFIGS["example-bad-gqa-fictional"]
        _, checks = amp_model_check.check_model("example-bad-gqa-fictional", cfg)
        gqa_check = [c for c in checks if "evenly divisible" in c[0]][0]
        self.assertFalse(gqa_check[1])


class TestHeadDimDerivation(unittest.TestCase):
    def test_mistral_head_dim_is_128(self):
        cfg = amp_model_check.MODEL_CONFIGS["mistral-7b-v0.1"]
        head_dim, _ = amp_model_check.check_model("mistral-7b-v0.1", cfg)
        self.assertEqual(head_dim, 128)


if __name__ == "__main__":
    unittest.main()
