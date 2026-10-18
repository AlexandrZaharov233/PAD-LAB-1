from pathlib import Path
import numpy as np
from scada_rag.common import DATA, ROOT, corpus_hash, digest, now, read_json, write_json
from scada_rag.preprocessing.text import chunk_document


INDEX_KEYS = ("embedding_model", "chunk_strategy", "chunk_size", "chunk_overlap")


def index_folder(cfg, docs):
    identity = {k: cfg[k] for k in INDEX_KEYS}
    identity.update({"corpus": corpus_hash(docs), "version": 3 if cfg["chunk_strategy"] == "paragraphs" else 2,
                     "revision": read_json(ROOT / "configs/model_revisions.json", {}).get(cfg["embedding_model"])})
    return DATA / "indexes" / digest(identity)[:20]


class VectorStore:
    def __init__(self, folder):
        import faiss
        self.folder = Path(folder)
        self.meta = read_json(self.folder / "metadata.json")
        if not self.meta or not self.meta.get("complete"):
            raise ValueError("Индекс не построен")
        # FAISS file APIs cannot reliably open Unicode Windows paths.
        self.index = faiss.deserialize_index(np.frombuffer((self.folder / "vectors.faiss").read_bytes(), dtype="uint8"))
        self.chunks = read_json(self.folder / "chunks.json")
        if self.index.ntotal != len(self.chunks):
            raise ValueError("Нарушена целостность индекса")

    @classmethod
    def build(cls, cfg, docs, embedder, log=print):
        import faiss
        folder = index_folder(cfg, docs)
        if read_json(folder / "metadata.json", {}).get("complete"):
            return cls(folder)
        chunks = [c for d in docs for c in chunk_document(d, cfg["chunk_strategy"], cfg["chunk_size"], cfg["chunk_overlap"])]
        if not chunks:
            raise ValueError("Нет документов. Сначала выполните grab")
        text = [c["title"] + ". " + c["section"] + "\n" + c["text"] for c in chunks]
        log(f"Векторизация: {len(chunks)} фрагментов, {cfg['embedding_model']}")
        vectors = np.ascontiguousarray(embedder.encode(text), dtype="float32")
        idx = faiss.IndexFlatIP(vectors.shape[1])
        idx.add(vectors)
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "metadata.json", {"complete": False})
        (folder / "vectors.faiss").write_bytes(faiss.serialize_index(idx).tobytes())
        write_json(folder / "chunks.json", chunks)
        write_json(folder / "metadata.json", {"complete": True, "created_at": now(), "corpus_hash": corpus_hash(docs),
                   "dimension": vectors.shape[1], "model_revision": embedder.revision,
                   "chunk_count": len(chunks), "config": {k: cfg[k] for k in INDEX_KEYS},
                   "truncated_chunks": embedder.truncation_count(text)})
        return cls(folder)

    def search(self, vector, k, source_id=None):
        if source_id:
            # Search all candidates before metadata filtering so matching documents
            # outside global Top-K are not silently lost.
            count = len(self.chunks)
        else:
            count = min(k, len(self.chunks))
        scores, ids = self.index.search(np.ascontiguousarray(vector, dtype="float32"), count)
        found = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            if source_id and chunk["source_id"] != source_id:
                continue
            found.append({**chunk, "similarity": float(score)})
            if len(found) == k:
                break
        return found
