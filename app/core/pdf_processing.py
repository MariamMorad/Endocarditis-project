"""
Extract PDF text using the Table of Contents (TOC) and merge pages into sections.
Same logic as the original notebook (Cells 5 & 6), unchanged.
"""
import re
import pymupdf

EXCLUDE_SECTION_KEYWORDS = [
    "appendix", "references", "bibliography", "acknowledgements",
    "table of contents", "list of tables", "list of figures",
    "author information", "supplementary data",
]


def extract_and_merge_pdf(pdf_path: str) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc()

    page_to_section = {}
    for item in toc:
        level, title, page = item
        if page not in page_to_section:
            page_to_section[page] = title

    merged_sections = {}
    current_section = "Front Matter"

    for page_num in range(len(doc)):
        actual_page = page_num + 1
        if actual_page in page_to_section:
            current_section = page_to_section[actual_page]

        page_text = doc.load_page(page_num).get_text("text")

        if current_section not in merged_sections:
            merged_sections[current_section] = {
                "source": pdf_path.split("/")[-1],
                "page": actual_page,
                "section": current_section,
                "text": "",
            }
        merged_sections[current_section]["text"] += "\n" + page_text

    final_data = []
    excluded_log = []
    for section_name, data in merged_sections.items():
        section_lower = section_name.lower()
        if any(kw in section_lower for kw in EXCLUDE_SECTION_KEYWORDS):
            excluded_log.append(section_name)
            continue

        text = data["text"]
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        text = re.sub(r'doi:\s*\S+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        cleaned_text = text.strip()

        if cleaned_text and len(cleaned_text) > 50:
            data["text"] = cleaned_text
            final_data.append(data)

    print(f"[{pdf_path.split('/')[-1]}] Kept {len(final_data)} sections, excluded {len(excluded_log)}")
    return final_data


def extract_all_pdfs(pdf_paths: list[str]) -> list[dict]:
    all_documents = []
    for pdf_path in pdf_paths:
        all_documents.extend(extract_and_merge_pdf(pdf_path))
    print(f"\nTotal merged, cleaned, non-reference sections: {len(all_documents)}")
    return all_documents
