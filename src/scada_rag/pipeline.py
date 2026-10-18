import time
from scada_rag.common import config, documents
from scada_rag.embeddings.models import Embedder
from scada_rag.retrieval.store import VectorStore
from scada_rag.reranking.ranker import Reranker, filter_hits
from scada_rag.generation.ollama import generate


class Pipeline:
    def __init__(self, cfg=None, log=print):
        self.cfg = config(cfg)
        self.docs = documents()
        if not self.docs:
            raise RuntimeError("Корпус пуст. Сначала запустите подготовку данных")
        self.embedder = Embedder(self.cfg["embedding_model"], self.cfg["cpu_threads"])
        self.store = VectorStore.build(self.cfg, self.docs, self.embedder, log)
        self.reranker = None

    def retrieve(self, question, overrides=None, query_vector=None):
        cfg = {**self.cfg, **(overrides or {})}
        started = time.perf_counter()
        vector = query_vector if query_vector is not None else self.embedder.encode([question], query=True)
        raw = self.store.search(vector, cfg["top_k"], cfg.get("source_id") or None)
        filtered = filter_hits(raw, cfg["similarity_threshold"])
        if cfg["use_reranker"] and filtered:
            if self.reranker is None:
                self.reranker = Reranker(cfg["reranker_model"])
            filtered = self.reranker.rank(question, filtered)
        context = filter_hits(filtered, cfg["similarity_threshold"], cfg["context_k"],
                              cfg["reranker_threshold"] if cfg["use_reranker"] else None)
        bounded, used = [], 0
        for hit in context:
            if used + len(hit["text"]) > cfg["context_chars"]:
                continue
            bounded.append(hit)
            used += len(hit["text"])
        return {"retrieved": raw, "context": bounded, "retrieval_seconds": time.perf_counter() - started}

    def ask(self, question, overrides=None):
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Введите вопрос длиной от 1 до 2000 символов")
        started = time.perf_counter()
        result = self.retrieve(question, overrides)
        answer = generate(question, result["context"], {**self.cfg, **(overrides or {})})
        return {**result, **answer, "total_seconds": time.perf_counter() - started}
