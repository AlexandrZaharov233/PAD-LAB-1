from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import tempfile

ROOT = Path(os.environ.get("SCADA_RAG_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Temporary file on the same filesystem: Windows also supports os.replace.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def config(overrides=None):
    result = read_json(ROOT / "configs/default.json")
    if overrides:
        result.update(overrides)
    if result["chunk_size"] < 100 or not 0 <= result["chunk_overlap"] < result["chunk_size"]:
        raise ValueError("Неверный размер или перекрытие фрагментов")
    if not 1 <= result["context_k"] <= 20 or not 1 <= result["top_k"] <= 100:
        raise ValueError("Неверное количество результатов")
    return result


def documents():
    return [read_json(p) for p in sorted((DATA / "processed").glob("*.json"))]


def corpus_hash(docs):
    # Include parsed content, not only bytes: OCR/parser changes invalidate indexes.
    return digest([{"id": d["id"], "blocks": d["blocks"], "url": d["url"]} for d in docs])
