import os
import sys
import shutil
import re
import subprocess
from pathlib import Path
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = BASE_DIR.parent / "The_Relay_Submission"

def main():
    print("=" * 80)
    print(" ASSEMBLING COMPLETE STANDALONE SUBMISSION FOLDER")
    print(f" Target Directory: {SUBMISSION_DIR}")
    print("=" * 80)

    if SUBMISSION_DIR.exists():
        shutil.rmtree(SUBMISSION_DIR)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Root files to copy
    root_files = [
        "REPORT.md",
        "results.csv",
        "make_tables.py",
        "ask.py",
        "chat_history.md",
        "README.md",
        "requirements.txt",
        ".env.example"
    ]

    for fname in root_files:
        src_path = BASE_DIR / fname
        if src_path.exists():
            shutil.copy2(src_path, SUBMISSION_DIR / fname)
            print(f" [OK] Copied {fname}")
        else:
            print(f" [WARN] File not found: {fname}")

    # 2. Copy src/ directory (excluding __pycache__)
    src_target = SUBMISSION_DIR / "src"
    src_target.mkdir(exist_ok=True)
    for py_file in (BASE_DIR / "src").glob("*.py"):
        shutil.copy2(py_file, src_target / py_file.name)
    print(" [OK] Copied src/ (all Python scripts, no cache)")

    # 3. Copy data/ directory
    data_target = SUBMISSION_DIR / "data"
    data_target.mkdir(exist_ok=True)
    required_data_files = [
        "paragraphs.jsonl",
        "chunks.jsonl",
        "benchmark_questions.json",
        "dense_embeddings_bge.npy",
        "corpus_v1_manifest.json",
        "retrieval_comparison_v1_manifest.json",
        "standin_verification_trace.json",
        "pipeline_trace_log.jsonl"
    ]
    for df in required_data_files:
        p = BASE_DIR / "data" / df
        if p.exists():
            shutil.copy2(p, data_target / df)
            print(f" [OK] Copied data/{df}")
        else:
            print(f" [WARN] Data file missing: data/{df}")

    # 4. Copy 3 Papers/ directory
    papers_src = BASE_DIR / "3 Papers"
    papers_target = SUBMISSION_DIR / "3 Papers"
    papers_target.mkdir(exist_ok=True)
    for pdf in papers_src.glob("*.pdf"):
        shutil.copy2(pdf, papers_target / pdf.name)
        print(f" [OK] Copied 3 Papers/{pdf.name}")

    # 5. Generate Styled HTML & Enforce STRICT 2-PAGE PDF Export
    report_md_path = SUBMISSION_DIR / "REPORT.md"
    report_html_path = SUBMISSION_DIR / "REPORT.html"
    report_pdf_path = SUBMISSION_DIR / "REPORT.pdf"

    if report_md_path.exists():
        with open(report_md_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        html_body = md_to_html(md_text)

        edge_paths = [
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Windows\SystemApps\Microsoft.MicrosoftEdge_8wekyb3d8bbwe\MicrosoftEdge.exe")
        ]
        edge_bin = next((p for p in edge_paths if p.exists()), None)

        # Style candidates to achieve EXACTLY 2 pages
        style_candidates = [
            {"font": "7.3pt", "lh": "1.14", "tbl_font": "6.5pt", "tbl_pad": "1.5px 3px", "h2_pad": "4px 0 1px 0", "p_pad": "1.5px 0"},
            {"font": "7.1pt", "lh": "1.12", "tbl_font": "6.3pt", "tbl_pad": "1.2px 2.5px", "h2_pad": "3px 0 1px 0", "p_pad": "1.2px 0"},
            {"font": "6.9pt", "lh": "1.10", "tbl_font": "6.1pt", "tbl_pad": "1px 2px", "h2_pad": "3px 0 1px 0", "p_pad": "1px 0"},
            {"font": "7.6pt", "lh": "1.16", "tbl_font": "6.7pt", "tbl_pad": "1.8px 3px", "h2_pad": "5px 0 2px 0", "p_pad": "2px 0"}
        ]

        pdf_pages = 0
        for cand in style_candidates:
            full_html = build_html(html_body, cand)
            with open(report_html_path, "w", encoding="utf-8") as f:
                f.write(full_html)

            if edge_bin:
                try:
                    cmd = [
                        str(edge_bin),
                        "--headless",
                        "--disable-gpu",
                        f"--print-to-pdf={report_pdf_path}",
                        str(report_html_path)
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if report_pdf_path.exists():
                        reader = PdfReader(str(report_pdf_path))
                        pdf_pages = len(reader.pages)
                        if pdf_pages == 2:
                            print(f" [OK] Generated REPORT.pdf: EXACTLY {pdf_pages} PAGES! (Fit verified, {report_pdf_path.stat().st_size // 1024} KB)")
                            break
                        else:
                            print(f" [INFO] Attempt produced {pdf_pages} pages; tuning styles...")
                except Exception as e:
                    print(f" [WARN] Edge print failed: {e}")
                    break

        if pdf_pages != 2 and report_pdf_path.exists():
            reader = PdfReader(str(report_pdf_path))
            print(f" [NOTICE] Final PDF page count: {len(reader.pages)}")

    # 6. Strict Security Audit of Target Folder
    print("\n" + "=" * 80)
    print(" RUNNING FINAL SAFETY AUDIT ON SUBMISSION FOLDER...")
    print("=" * 80)

    has_env = (SUBMISSION_DIR / ".env").exists()
    has_venv = (SUBMISSION_DIR / "venv").exists()

    leak_found = False
    for root, dirs, files in os.walk(SUBMISSION_DIR):
        for file in files:
            p = Path(root) / file
            if p.suffix in [".py", ".md", ".jsonl", ".csv", ".txt", ".example"]:
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        matches = re.findall(r"gsk_[A-Za-z0-9]{20,}", content)
                        if matches:
                            print(f" [CRITICAL WARNING] Potential API key found in: {p.relative_to(SUBMISSION_DIR)}")
                            leak_found = True
                except Exception:
                    pass

    print(f" .env file excluded:       {'PASS [OK]' if not has_env else 'FAIL [DANGER]'}")
    print(f" venv/ directory excluded: {'PASS [OK]' if not has_venv else 'FAIL [DANGER]'}")
    print(f" API Key search:           {'PASS [OK] No keys detected' if not leak_found else 'FAIL [LEAK FOUND]'}")

    print("\n" + "=" * 80)
    print(" SUBMISSION FOLDER READY TO UPLOAD!")
    print(f" Full path: {SUBMISSION_DIR}")
    print(" You can now upload this entire folder directly to Google Drive.")
    print("=" * 80)

def build_html(body: str, style: dict) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>The Relay: Resilient RAG Pipeline & Load Study</title>
<style>
  @page {{
    size: A4;
    margin: 8mm 9mm 8mm 9mm;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: {style['font']};
    line-height: {style['lh']};
    color: #1a1e22;
    margin: 0;
    padding: 0;
  }}
  h1 {{
    font-size: 11pt;
    margin: 0 0 4px 0;
    padding-bottom: 2px;
    border-bottom: 1.2px solid #eaecef;
    color: #0b1014;
    text-align: center;
  }}
  h2 {{
    font-size: 8.5pt;
    font-weight: 700;
    margin: {style['h2_pad']};
    padding-bottom: 1px;
    border-bottom: 1px solid #eaecef;
    color: #0b1014;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 7.8pt;
    font-weight: 600;
    margin: 3px 0 1px 0;
    color: #24292e;
    page-break-after: avoid;
  }}
  p {{
    margin: {style['p_pad']};
  }}
  ul, ol {{
    margin: 1.5px 0;
    padding-left: 14px;
  }}
  li {{
    margin: 0.5px 0;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 2px 0;
    font-size: {style['tbl_font']};
    page-break-inside: avoid;
    table-layout: auto;
  }}
  th, td {{
    border: 0.8px solid #c8ccd0;
    padding: {style['tbl_pad']};
    text-align: left;
    vertical-align: middle;
  }}
  th {{
    background-color: #f6f8fa;
    font-weight: 600;
  }}
  code {{
    font-family: Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 7.2pt;
    background-color: rgba(27,31,35,0.05);
    padding: 0.5px 2px;
    border-radius: 2px;
  }}
  hr {{
    border: 0;
    height: 0.8px;
    background: #e1e4e8;
    margin: 3px 0;
  }}
</style>
</head>
<body>
{body}
</body>
</html>"""

def md_to_html(md: str) -> str:
    lines = md.split("\n")
    html_lines = []
    in_table = False
    table_header_done = False

    for line in lines:
        s = line.strip()
        if not s:
            if in_table:
                html_lines.append("</table>")
                in_table = False
                table_header_done = False
            continue

        if s.startswith("# "):
            html_lines.append(f"<h1>{format_inline(s[2:])}</h1>")
        elif s.startswith("## "):
            html_lines.append(f"<h2>{format_inline(s[3:])}</h2>")
        elif s.startswith("### "):
            html_lines.append(f"<h3>{format_inline(s[4:])}</h3>")
        elif s.startswith("---"):
            html_lines.append("<hr>")
        elif s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s[1:-1].split("|")]
            if all(set(c).issubset({":", "-", " "}) for c in cells):
                table_header_done = True
                continue
            if not in_table:
                in_table = True
                html_lines.append("<table>")
                tag = "th"
            else:
                tag = "td" if table_header_done else "th"
            row_html = "".join(f"<{tag}>{format_inline(c)}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row_html}</tr>")
        elif s.startswith("- "):
            html_lines.append(f"<ul><li>{format_inline(s[2:])}</li></ul>")
        elif s.startswith("1. ") or s.startswith("2. ") or s.startswith("3. ") or s.startswith("4. "):
            html_lines.append(f"<ol><li>{format_inline(s[3:])}</li></ol>")
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False
                table_header_done = False
            html_lines.append(f"<p>{format_inline(s)}</p>")

    if in_table:
        html_lines.append("</table>")

    return "\n".join(html_lines)

def format_inline(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)
    text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)
    return text

if __name__ == "__main__":
    main()
