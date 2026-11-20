"""Compile README + reports/01–07 into one PDF (markdown -> HTML -> headless Chrome)."""
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
parts = [ROOT / "README.md"] + sorted((ROOT / "reports").glob("0*.md"))
body = "\n".join(
    "<section>" + markdown.markdown(re.sub(r"(?m)^ {1,3}(?=[-*] )", "    ", p.read_text()),
                                    extensions=["tables", "fenced_code"]) + "</section>"
    for p in parts)
css = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.5; color: #1c2430; }
section { page-break-before: always; } section:first-child { page-break-before: avoid; }
h1 { font-size: 20pt; color: #0b3d2e; border-bottom: 2px solid #21a038; padding-bottom: 4px; }
h2 { font-size: 13.5pt; color: #0b3d2e; margin-top: 1.4em; }
h3 { font-size: 11.5pt; }
table { border-collapse: collapse; margin: .6em 0 1em; font-size: 9.5pt; width: 100%; }
th, td { border: 1px solid #d5dde3; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #eef5f0; } tr { page-break-inside: avoid; }
blockquote { margin: .6em 0; padding: .5em 1em; background: #f4f8f5; border-left: 3px solid #21a038; }
code, pre { font-family: Menlo, monospace; font-size: 8.8pt; background: #f3f5f7; }
pre { padding: 8px; white-space: pre-wrap; }
img { max-width: 85%; display: block; margin: .8em auto; }
"""
html = ROOT / "reports" / "_full_report.html"
html.write_text(f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>")
pdf = ROOT / "Sber_Autosubscription_Project_Report.pdf"
subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless",
                "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", html.as_uri()],
               check=True, capture_output=True)
html.unlink()
print(pdf)
