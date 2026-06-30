#!/usr/bin/env python3
"""AMP Spaces: paste a CUDA/HIP kernel, get a CUDA-to-ROCm diagnosis and
mechanical fix instantly, in the browser.

Run:
    pip install -r web/requirements.txt
    python3 web/app.py
"""
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
from openai import OpenAI

# Works both locally (web/app.py -> ../scripts) and in HF Space (app.py -> ./scripts)
_here = Path(__file__).parent
_scripts = _here / "scripts" if (_here / "scripts").exists() else _here.parent / "scripts"
sys.path.insert(0, str(_scripts))
import amp_diagnose  # noqa: E402
import amp_suggest_fix  # noqa: E402

_root = _here if (_here / "tests").exists() else _here.parent
EXAMPLE_BUGGY_KERNEL = (_root / "tests" / "fixtures" / "buggy_matmul_fixture.cuh").read_text(encoding="utf-8")

SEVERITY_BADGE = {"high": "🔴 HIGH", "info": "🔵 INFO"}

_fw_client = None


def _get_fw_client():
    global _fw_client
    if _fw_client is None:
        key = os.environ.get("FIREWORKS_API_KEY", "")
        if key:
            _fw_client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=key,
            )
    return _fw_client


def ai_explain(kernel_code: str, findings_text: str) -> str:
    client = _get_fw_client()
    if client is None:
        return "_AI explanation unavailable: FIREWORKS_API_KEY not set._"

    prompt = (
        "You are an expert in CUDA-to-ROCm/HIP GPU kernel migration.\n\n"
        "A deterministic static analysis tool found the following issues "
        "in this kernel:\n\n"
        f"FINDINGS:\n{findings_text}\n\n"
        f"KERNEL (first 2000 chars):\n{kernel_code[:2000]}\n\n"
        "For each finding explain:\n"
        "1. What it means in plain English\n"
        "2. Why it causes incorrect results specifically on ROCm/AMD hardware\n"
        "3. What to verify after applying the suggested fix\n\n"
        "Be concise. Do not rewrite the kernel. Focus on WHY, not HOW."
    )

    resp = client.chat.completions.create(
        model="accounts/fireworks/models/deepseek-v4-pro",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
        temperature=0.2,
    )
    return resp.choices[0].message.content


def diagnose(code: str, tile: str):
    if not code.strip():
        return "Paste a kernel first.", "", "", ""

    tile_values = amp_diagnose.parse_tile_arg(tile.strip()) if tile.strip() else None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cuh", delete=False,
                                     encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name

    try:
        patterns = amp_diagnose.load_patterns()
        findings = amp_diagnose.analyze_file(tmp_path, tile_values)

        if not findings:
            return "✅ No suspicious patterns found by static analysis.", "", "", ""

        report_lines = []
        for fnd in findings:
            p = patterns.get(fnd.pattern_id, {})
            sev = p.get("severity", "info")
            report_lines.append(
                f"**{SEVERITY_BADGE.get(sev, sev.upper())}** "
                f"line {fnd.line}: {p.get('name', fnd.pattern_id)}\n\n"
                f"> `{fnd.code}`\n\n"
                f"{fnd.extra or p.get('description', '')}\n"
            )

        results = amp_suggest_fix.suggest([tmp_path], tile_values)
        r = results[0]

        diff_text = r["diff"] or "(no mechanical fix available for these findings)"
        fixed_code = r["new_text"] if r["new_text"] else code

        if r["unfixable_findings"]:
            report_lines.append("\n---\n**No mechanical fix, explanation only:**\n")
            for u in r["unfixable_findings"]:
                report_lines.append(
                    f"- line {u['line']} ({u['pattern_id']}): {u['description']} "
                    f"(*{u['suggestion']}*)"
                )

        plain_findings = "\n".join(
            f"line {fnd.line}: {fnd.pattern_id} -- {fnd.code}"
            for fnd in findings
        )
        ai_text = ai_explain(code, plain_findings)

        return "\n".join(report_lines), diff_text, fixed_code, ai_text
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def save_fixed(fixed_code: str):
    if not fixed_code.strip():
        return None
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".cuh",
                                      delete=False, encoding="utf-8",
                                      prefix="amp_fixed_")
    tmp.write(fixed_code)
    tmp.close()
    return tmp.name


with gr.Blocks(title="AMP: CUDA-ROCm Parity Check") as demo:
    gr.Markdown(
        "# 🛠️ AMP: CUDA-ROCm Parity Check\n"
        "### Instant CUDA to ROCm kernel diagnosis, powered by Fireworks AI\n"
        "**New here?** A buggy kernel is already loaded. Click **Diagnose** to run.\n\n"
        "Have your own kernel? Click **Clear**, paste it in, then click **Diagnose**. "
        "Full workflow: [github.com/agusbass/AMP](https://github.com/agusbass/AMP)."
    )

    with gr.Row():
        # --- LEFT: input ---
        with gr.Column(scale=1):
            code_in = gr.Code(
                label="Your kernel (.cu / .cuh)",
                language="cpp",
                value=EXAMPLE_BUGGY_KERNEL,
                lines=22,
            )
            with gr.Row():
                clear_btn = gr.Button("Clear (paste your own)", variant="secondary", size="sm")
                tile_in = gr.Textbox(label="Tile (BM,BN,BK)", value="32,16,32", scale=1)
            btn = gr.Button("Diagnose", variant="primary")

        # --- RIGHT: output ---
        with gr.Column(scale=1):
            findings_out = gr.Markdown("_Click Diagnose to see findings._")
            diff_out = gr.Textbox(
                label="Suggested fix (unified diff)",
                lines=8,
                interactive=False,
            )
            with gr.Row():
                fixed_out = gr.Textbox(
                    label="Fixed kernel (select all + Ctrl+C to copy)",
                    lines=12,
                    interactive=False,
                )
            download_btn = gr.DownloadButton("Download fixed kernel (.cuh)", visible=False)

    with gr.Accordion("AI Explanation (powered by Fireworks AI / Llama-3-70B)", open=True):
        ai_out = gr.Markdown("_AI explanation will appear here after diagnosis._")

    # wire up buttons
    clear_btn.click(lambda: "", outputs=[code_in])

    btn.click(
        diagnose,
        inputs=[code_in, tile_in],
        outputs=[findings_out, diff_out, fixed_out, ai_out],
    ).then(
        save_fixed,
        inputs=[fixed_out],
        outputs=[download_btn],
    ).then(
        lambda code: gr.DownloadButton(visible=bool(code.strip())),
        inputs=[fixed_out],
        outputs=[download_btn],
    )

if __name__ == "__main__":
    demo.launch()
