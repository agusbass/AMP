"""Tests for scripts/amp_generate_wrapper.py -- no GPU needed."""
import subprocess
import sys
import tempfile
import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = REPO_ROOT / "scripts" / "amp_generate_wrapper.py"


def run_generator(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, check=True,
    )


class TestGeneratedWrapperShape:
    def test_exports_the_symbol_amp_validate_kernel_looks_for(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "w.cu")
            run_generator("my_kernel", "--output", out)
            content = open(out).read()
            assert 'extern "C" int amp_user_gemm_fp32(' in content

    def test_forward_declares_the_named_kernel(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "w.cu")
            run_generator("super_special_gemm", "--output", out)
            content = open(out).read()
            assert "super_special_gemm" in content
            assert "super_special_gemm<<<grid, block>>>" in content

    def test_compiles_unmodified_under_both_hipcc_and_nvcc_branches(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "w.cu")
            run_generator("k", "--output", out)
            content = open(out).read()
            assert "#if defined(__HIPCC__)" in content
            assert "#include <hip/hip_runtime.h>" in content
            assert "#include <cuda_runtime.h>" in content

    def test_custom_block_dims_applied(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "w.cu")
            run_generator("k", "--output", out, "--block", "32,8")
            content = open(out).read()
            assert "dim3 block(32, 8)" in content
