"""
Split each section into fixed-size chunks with overlap. chunk_id is built from a hash
of the full section name (instead of section[:20]) to avoid collisions between
similarly-named sections.
"""
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_merged_sections(all_documents: list[dict], chunk_size: int = 900, chunk_overlap: int = 150) -> list[dict]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    final_chunks = []
    for doc in all_documents:
        section_chunks = text_splitter.split_text(doc["text"])
        section_hash = hashlib.md5(doc["section"].encode()).hexdigest()[:8]
        for i, chunk_text in enumerate(section_chunks):
            chunk_data = {
                "source": doc["source"],
                "section": doc["section"],
                "section_start_page": doc["page"],
                "chunk_id": f"{doc['source']}_Sec_{section_hash}_Chunk_{i + 1}",
                "text": chunk_text,
                "text_hash": hashlib.md5(chunk_text.strip().lower().encode()).hexdigest(),
            }
            final_chunks.append(chunk_data)

    print(f"Total raw chunks before dedup: {len(final_chunks)}")
    return final_chunks
