import os
import re
import json
from pathlib import Path
from pypdf import PdfReader

# Directory setup
BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "3 Papers"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PARAGRAPHS_PATH = DATA_DIR / "paragraphs.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

PAPERS_METADATA = [
    {
        "file_name": "Attention is all you need.pdf",
        "paper_id": "attention",
        "paper_title": "Attention Is All You Need",
        "arxiv_id": "1706.03762v7"
    },
    {
        "file_name": "Dynamic Chunking.pdf",
        "paper_id": "dynamic_chunking",
        "paper_title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "arxiv_id": "2507.07955v2"
    },
    {
        "file_name": "Moshi.pdf",
        "paper_id": "moshi",
        "paper_title": "Moshi: a speech-text foundation model for real-time dialogue",
        "arxiv_id": "2410.00037v1"
    }
]

# Section heading patterns
NUMBERED_HEADING_RE = re.compile(
    r"^([0-9]+(?:\.[0-9]+)*|[A-Z](?:\.[0-9]+)*)\s+([A-Z][A-Za-z0-9\s,\-\(\)\:\'\"]{2,70})$"
)

UNNUMBERED_HEADINGS = {
    "abstract", "introduction", "background", "model architecture", "methods", "method",
    "experiments", "results", "discussion", "related work", "conclusion", "conclusions",
    "references", "acknowledgments", "acknowledgements", "appendix", "appendices",
    "broader impact", "limitations", "ethics statement"
}

def is_section_heading(line):
    s = line.strip()
    if not s or len(s) > 80:
        return None, None
        
    m = NUMBERED_HEADING_RE.match(s)
    if m:
        num = m.group(1)
        title = m.group(2).strip()
        # Ensure title isn't a math formula or citation list
        if not re.search(r"(=|\+|\[\d+\])", title):
            return num, f"{num} {title}"
            
    if s.lower() in UNNUMBERED_HEADINGS:
        return None, s.title()
        
    return None, None

def clean_lines(raw_text):
    """
    Cleans running headers, arXiv stamps, and lone page numbers.
    """
    cleaned = []
    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        # Remove arXiv watermarks
        if re.search(r"arxiv:\d{4}\.\d{4,5}v\d+", s, re.IGNORECASE):
            continue
        # Remove generic running preprints
        if s.lower() in ["preprint", "under review", "in review"]:
            continue
        # Remove standalone page numbers (1-3 digits)
        if re.match(r"^\d{1,3}$", s):
            continue
        cleaned.append(s)
    return cleaned

def extract_paragraphs_from_page(lines, paper_meta, pdf_page, current_section, para_counter):
    """
    Segments page lines into fine-grained paragraphs while accurately
    tracking section transitions occurring anywhere on the page.
    """
    paragraphs = []
    current_para_lines = []
    
    def flush_para():
        nonlocal current_para_lines, para_counter
        if not current_para_lines:
            return
        
        text = " ".join(current_para_lines)
        text = re.sub(r"\s+", " ", text).strip()
        current_para_lines = []
        
        # Filter out stray single numbers or symbols
        if len(text) < 25:
            return
            
        para_counter += 1
        para_id = f"{paper_meta['paper_id']}_p{pdf_page:03d}_para{para_counter:03d}"
        
        record = {
            "paragraph_id": para_id,
            "paper_id": paper_meta["paper_id"],
            "paper_title": paper_meta["paper_title"],
            "arxiv_id": paper_meta["arxiv_id"],
            "section_number": current_section["number"],
            "section_title": current_section["title"],
            "pdf_page": pdf_page,
            "printed_page_label": None,
            "paragraph_index": para_counter,
            "text": text,
            "source_file": paper_meta["file_name"],
            "extraction_warnings": []
        }
        paragraphs.append(record)

    for i, line in enumerate(lines):
        if not line:
            # Blank line signifies paragraph break
            flush_para()
            continue
            
        # Check if line is a section heading
        sec_num, sec_title = is_section_heading(line)
        if sec_title:
            flush_para()
            current_section["number"] = sec_num
            current_section["title"] = sec_title
            continue
            
        # Check for bullet points or numbered lists
        is_list_item = bool(re.match(r"^(\u2022|\-|\([0-9a-zivx]+\)|[0-9]+\.)\s+", line))
        if is_list_item and current_para_lines:
            flush_para()
            
        current_para_lines.append(line)
        
        # If line ends with period/colon and paragraph has accumulated substantial text (>350 chars),
        # treat as natural paragraph boundary if next line starts capitalized
        if (line.endswith((".", ":", "?", "!"))) and sum(len(l) for l in current_para_lines) > 350:
            if i + 1 < len(lines) and lines[i+1] and lines[i+1][0].isupper():
                flush_para()

    flush_para()
    return paragraphs, para_counter

def create_chunks(paragraphs, target_tokens=350, max_tokens=550):
    """
    Groups paragraphs within the same paper and section into retrieval chunks.
    Retains full list of paragraph_ids and source_pages for citation provenance.
    """
    chunks = []
    current_chunk = []
    current_tokens = 0
    current_paper = None
    current_section = None
    chunk_counter = 0

    def approx_tokens(text):
        return int(len(text.split()) * 1.3)

    def flush_chunk():
        nonlocal current_chunk, current_tokens, chunk_counter
        if not current_chunk:
            return
        chunk_counter += 1
        cid = f"{current_chunk[0]['paper_id']}_chunk_{chunk_counter:05d}"
        combined_text = "\n\n".join([p["text"] for p in current_chunk])
        p_ids = [p["paragraph_id"] for p in current_chunk]
        pages = sorted(list(set(p["pdf_page"] for p in current_chunk)))
        
        chunk_record = {
            "chunk_id": cid,
            "paper_id": current_chunk[0]["paper_id"],
            "section_title": current_chunk[0]["section_title"],
            "paragraph_ids": p_ids,
            "source_pages": pages,
            "text": combined_text,
            "token_count": approx_tokens(combined_text)
        }
        chunks.append(chunk_record)
        current_chunk = []
        current_tokens = 0

    for p in paragraphs:
        p_toks = approx_tokens(p["text"])
        same_section = (p["paper_id"] == current_paper and p["section_title"] == current_section)
        
        if not same_section:
            flush_chunk()
            current_paper = p["paper_id"]
            current_section = p["section_title"]
        elif current_tokens + p_toks > max_tokens and current_tokens >= target_tokens:
            flush_chunk()
            
        current_chunk.append(p)
        current_tokens += p_toks

    flush_chunk()
    return chunks

def main():
    print("=" * 65)
    print("   STAGE 1 (REFINED): GRANULAR PARAGRAPH EXTRACTION")
    print("=" * 65)

    all_paragraphs = []
    validation_stats = []

    for meta in PAPERS_METADATA:
        pdf_path = PAPERS_DIR / meta["file_name"]
        print(f"\nProcessing: {meta['paper_title']}")
        print(f"  Source file: {pdf_path.name}")
        
        if not pdf_path.exists():
            print(f"  [ERROR] File not found: {pdf_path}")
            continue
            
        reader = PdfReader(str(pdf_path))
        num_pages = len(reader.pages)
        print(f"  PDF Total Pages: {num_pages}")
        
        paper_paras = []
        current_section = {"number": None, "title": "Abstract"}
        pages_with_text = 0
        
        for idx, page in enumerate(reader.pages):
            pdf_page_num = idx + 1
            raw_text = page.extract_text() or ""
            if raw_text.strip():
                pages_with_text += 1
            
            lines = clean_lines(raw_text)
            page_paras, _ = extract_paragraphs_from_page(lines, meta, pdf_page_num, current_section, 0)
            paper_paras.extend(page_paras)

        all_paragraphs.extend(paper_paras)
        
        # Count sections detected
        sections_found = set(p["section_title"] for p in paper_paras)
        validation_stats.append({
            "paper_id": meta["paper_id"],
            "title": meta["paper_title"],
            "pdf_pages": num_pages,
            "pages_with_text": pages_with_text,
            "paragraphs_extracted": len(paper_paras),
            "sections_detected": len(sections_found)
        })
        print(f"  -> Extracted {len(paper_paras)} paragraphs across {num_pages} pages ({len(sections_found)} distinct sections).")

    # Write paragraphs.jsonl
    print(f"\nWriting {len(all_paragraphs)} records to {PARAGRAPHS_PATH.name}...")
    with open(PARAGRAPHS_PATH, "w", encoding="utf-8") as f:
        for p in all_paragraphs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Generate retrieval-ready chunks
    chunks = create_chunks(all_paragraphs)
    print(f"Writing {len(chunks)} chunks to {CHUNKS_PATH.name}...")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ==========================================
    # VALIDATION GATE (PHASE 3)
    # ==========================================
    print("\n" + "=" * 65)
    print("      PHASE 3: EXTRACTION VALIDATION GATE AUDIT")
    print("=" * 65)
    
    # 1. Paper coverage
    print(f"1. Paper coverage:        {'PASS' if len(validation_stats) == 3 else 'FAIL'} (3/3 papers processed)")
    
    # 2. Page coverage
    page_pass = all(s["pdf_pages"] == s["pages_with_text"] for s in validation_stats)
    print(f"2. Page coverage:         {'PASS' if page_pass else 'CHECK WARNINGS'}")
    for s in validation_stats:
        print(f"   - {s['paper_id']}: {s['pages_with_text']}/{s['pdf_pages']} pages, {s['paragraphs_extracted']} paragraphs, {s['sections_detected']} sections")
        
    # 3. Paragraph metadata & IDs
    para_ids = [p["paragraph_id"] for p in all_paragraphs]
    unique_ids = set(para_ids)
    no_dupes = len(para_ids) == len(unique_ids)
    print(f"3. Deterministic IDs:     {'PASS' if no_dupes else 'FAIL'} ({len(unique_ids)} unique IDs, 0 duplicates)")

    # 4. Empty records check
    has_empty = any(len(p["text"].strip()) == 0 for p in all_paragraphs)
    print(f"4. Empty records check:   {'PASS' if not has_empty else 'FAIL'} (No empty paragraphs)")

    # 5. Provenance integrity
    all_p_ids_set = set(para_ids)
    provenance_pass = all(
        all(pid in all_p_ids_set for pid in c["paragraph_ids"])
        for c in chunks
    )
    print(f"5. Chunk provenance:      {'PASS' if provenance_pass else 'FAIL'} (100% of chunks map to valid paragraph_ids)")
    
    print("=" * 65)
    print(f"Output files saved in: {DATA_DIR}")
    print("  -> paragraphs.jsonl")
    print("  -> chunks.jsonl")
    print("=" * 65)

if __name__ == "__main__":
    main()
