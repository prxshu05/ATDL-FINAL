#!/usr/bin/env python3
"""Build the report: report.md -> report.html -> report.pdf (Chrome), and an Overleaf-ready LaTeX source in latex/."""
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
src = open("report.md").read().replace("{{ALL_RUNS}}", open("all_runs.md").read())
open("_full.md", "w").write(src)

# ---------------------------------------------------------------- HTML + PDF
subprocess.run(["pandoc", "_full.md", "-s", "--mathml", "--number-sections", "--css", "style.css", "-o", "report.html"],
               check=True)
subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=6000",
                "--print-to-pdf=report.pdf", f"file://{HERE}/report.html"], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)

# ---------------------------------------------------------------- LaTeX
tex = src
tex = re.sub(r'::: \{\.eq n="(\d+)"\}\n\$\$(.*?)\$\$\n:::',
             lambda m: "```{=latex}\n\\begin{equation}" + m.group(2).strip().rstrip(",") + "\\label{eq:" + m.group(1)
             + "}\\end{equation}\n```", tex, flags=re.S)
tex = tex.replace("::: {.refs}\n", "").replace("\n:::\n\n# Appendix A", "\n\n# Appendix A")
tex = tex.replace("p̄_c", "$\\bar p_c$")
tex = tex.replace(" · ", ", ")
for a, b in [("−", "$-$"), ("×", "$\\times$"), ("±", "$\\pm$"), ("→", "$\\to$"), ("λ", "$\\lambda$"), ("≈", "$\\approx$"),
             ("Δ", "$\\Delta$"), ("≥", "$\\ge$"), ("≤", "$\\le$"), ("²", "$^2$"), ("·", "$\\cdot$"), ("τ", "$\\tau$"),
             ("∈", "$\\in$"), ("ρ", "$\\rho$"), ("…", "\\ldots{}")]:
    tex = tex.replace(a, b)
tex = re.sub(r"(figs/[a-z0-9_]+)\.png", r"\1.pdf", tex)
# drop the HTML-only algorithm block before checking for leftover unicode
tex_noh = re.sub(r"```\{=html\}.*?```", "", tex, flags=re.S)
left = {c for c in tex_noh if ord(c) > 127 and c not in "–—“”’‘"}
if left:
    print("WARNING: unicode left for LaTeX:", left)
open("_tex.md", "w").write(tex_noh)
header = r"""
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{xurl}
\usepackage{fancyhdr}
\pagestyle{fancy}\fancyhf{}\renewcommand{\headrulewidth}{0pt}
\fancyhead[C]{\footnotesize\itshape Query-Efficient Black-Box Knowledge Distillation from Pure Noise}
\fancyfoot[C]{\thepage}
\setlength{\LTpre}{6pt}\setlength{\LTpost}{6pt}
"""
open("_header.tex", "w").write(header)
os.makedirs("latex/figs", exist_ok=True)
for f in os.listdir("figs"):
    if f.endswith(".pdf"):
        shutil.copy(f"figs/{f}", f"latex/figs/{f}")
subprocess.run(["pandoc", "_tex.md", "-s", "--number-sections", "-V", "documentclass=article", "-V", "fontsize=10pt",
                "-V", "geometry:margin=2.4cm", "-V", "fontfamily=libertine", "-V", "colorlinks=true", "-V", "linkcolor=blue",
                "-V", "urlcolor=blue", "--include-in-header=_header.tex", "-o", "latex/report.tex"], check=True)
for f in ("_full.md", "_tex.md", "_header.tex"):
    os.remove(f)
shutil.copy("report.pdf", "latex/report_compiled_preview.pdf")
if os.path.exists("ATDL_report_overleaf.zip"):
    os.remove("ATDL_report_overleaf.zip")
subprocess.run(["zip", "-qr", "../ATDL_report_overleaf.zip", "report.tex", "figs"], cwd="latex", check=True)
pages = len(re.findall(rb"/Type\s*/Page[^s]", open("report.pdf", "rb").read()))
print(f"built report.pdf ({pages} pages), report.html, latex/report.tex, ATDL_report_overleaf.zip")
