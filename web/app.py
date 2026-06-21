#!/usr/bin/env python3
"""AMP Spaces — paste a CUDA/HIP kernel, get a CUDA->AMD diagnosis and
mechanical fix instantly, in the browser.

This is a UI shell around the exact same tested logic CLI users run
(scripts/amp_diagnose.py's analyze_file, scripts/amp_suggest_fix.py's
suggest) -- no new diagnosis logic lives here, only presentation. The
two pasted-code paths write to a temp file and call the same functions
tests/test_diagnose.py and tests/test_suggest_fix.py already cover.

Run:
    pip install -r web/requirements.txt
    python3 web/app.py
"""
import sys
import tempfile
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import amp_diagnose  # noqa: E402
import amp_suggest_fix  # noqa: E402

EXAMPLE_BUGGY_KERNEL = (Path(__file__).parent.parent
                         / "tests" / "fixtures" / "buggy_matmul_fixture.cuh").read_text(encoding="utf-8")

SEVERITY_BADGE = {"high": "🔴 HIGH", "info": "🔵 INFO"}


def diagnose(code: str, tile: str):
    if not code.strip():
        return "Paste a kernel first.", "", code

    tile_values = amp_diagnose.parse_tile_arg(tile.strip()) if tile.strip() else None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cuh", delete=False,
                                      encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name

    try:
        patterns = amp_diagnose.load_patterns()
        findings = amp_diagnose.analyze_file(tmp_path, tile_values)

        if not findings:
            return ("✅ No suspicious patterns found by static analysis.",
                    "", code)

        report_lines = []
        for fnd in findings:
            p = patterns.get(fnd.pattern_id, {})
            sev = p.get("severity", "info")
            report_lines.append(
                f"**{SEVERITY_BADGE.get(sev, sev.upper())}** "
                f"line {fnd.line} — {p.get('name', fnd.pattern_id)}\n\n"
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
                    f"— *{u['suggestion']}*"
                )

        return "\n".join(report_lines), diff_text, fixed_code
    finally:
        Path(tmp_path).unlink(missing_ok=True)


with gr.Blocks(title="AMP — CUDA to AMD kernel diagnosis") as demo:
    gr.Markdown(
        "# 🛠️ AMP — instant CUDA→AMD kernel diagnosis\n"
        "Paste a GPU kernel, click **Diagnose for AMD**. Static analysis only "
        "— no GPU, no compiler needed. This UI calls the exact same "
        "`amp_diagnose.py`/`amp_suggest_fix.py` logic the CLI and "
        "`parity_check.py --analyze` use on a real cross-vendor FAIL. "
        "See [the repo](https://github.com/agusbass/AMP) for the full "
        "validated CUDA↔HIP parity-check workflow (this Space covers the "
        "no-GPU diagnosis half of it)."
    )
    with gr.Row():
        with gr.Column():
            code_in = gr.Code(label="Your kernel (.cu/.cuh)", language="cpp",
                               value=EXAMPLE_BUGGY_KERNEL, lines=20)
            tile_in = gr.Textbox(label="Tile config (BM,BN,BK) — optional",
                                  value="32,16,32")
            btn = gr.Button("Diagnose for AMD", variant="primary")
        with gr.Column():
            findings_out = gr.Markdown(label="Findings")
            diff_out = gr.Code(label="Suggested fix (unified diff)", language="diff")
            fixed_out = gr.Code(label="Fixed kernel", language="cpp")

    btn.click(diagnose, inputs=[code_in, tile_in],
              outputs=[findings_out, diff_out, fixed_out])

if __name__ == "__main__":
    demo.launch()
