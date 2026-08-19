"""
Remove exact-duplicate chunks and near-duplicate chunks per source document.
"""
from difflib import SequenceMatcher


def deduplicate_chunks(chunks: list[dict], near_dup_threshold: float = 0.92) -> list[dict]:
    seen_hashes = set()
    exact_deduped = []
    for c in chunks:
        if c["text_hash"] not in seen_hashes:
            seen_hashes.add(c["text_hash"])
            exact_deduped.append(c)

    print(f"Removed {len(chunks) - len(exact_deduped)} exact duplicates")

    final = []
    kept_texts_by_source: dict[str, list[str]] = {}
    near_dup_removed = 0

    for c in exact_deduped:
        src = c["source"]
        kept_texts_by_source.setdefault(src, [])
        is_dup = False
        for prev_text in kept_texts_by_source[src]:
            ratio = SequenceMatcher(None, c["text"][:300], prev_text[:300]).ratio()
            if ratio >= near_dup_threshold:
                is_dup = True
                near_dup_removed += 1
                break
        if not is_dup:
            kept_texts_by_source[src].append(c["text"])
            final.append(c)

    print(f"Removed {near_dup_removed} near-duplicates")
    print(f"Final deduplicated chunk count: {len(final)}")
    return final
