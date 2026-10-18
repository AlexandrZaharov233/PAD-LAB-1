import re
import unicodedata
from scada_rag.common import digest


def clean(text):
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "").replace("\u00a0", " ")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def windows(text, size, overlap, boundary=False):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Размер должен быть больше перекрытия")
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if boundary and end < len(text):
            options = [text.rfind(sep, start + size // 2, end) for sep in ("\n", ". ", " ")]
            cut = max(options)
            if cut > start:
                end = cut + 1
        part = text[start:end].strip()
        if part:
            yield part
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def chunk_document(doc, strategy="sections", size=1000, overlap=150):
    if strategy not in {"fixed", "overlap", "paragraphs", "sections"}:
        raise ValueError("Неизвестная стратегия")
    groups = []
    for block in doc["blocks"]:
        key = (block.get("section", ""), block.get("page"))
        # Sections aggregate consecutive paragraphs; fixed windows ignore headings,
        # but page borders are preserved for accurate PDF citations.
        merge = groups and groups[-1][0][1] == key[1] and strategy != "paragraphs"
        if strategy == "sections":
            merge = merge and groups[-1][0] == key
        if merge:
            groups[-1][1] += "\n\n" + clean(block["text"])
        else:
            groups.append([key, clean(block["text"])])
    chunks = []
    seen = set()
    for (section, page), text in groups:
        step_overlap = overlap if strategy in ("overlap", "sections") else 0
        paragraphs = re.split(r"\n\s*\n", text) if strategy == "paragraphs" else [text]
        for paragraph in paragraphs:
            for part in windows(paragraph, size, step_overlap, boundary=strategy in ("sections", "paragraphs")):
                if len(part) < 40 or part in seen:
                    continue
                seen.add(part)
                chunks.append({"id": digest([doc["id"], section, page, part])[:24], "source_id": doc["id"],
                               "source": doc["source"], "url": doc["url"], "title": doc["title"], "section": section,
                               "page": page, "text": part, "language": doc["language"], "note": doc.get("note", ""),
                               "updated_at": doc["updated_at"]})
    return chunks
