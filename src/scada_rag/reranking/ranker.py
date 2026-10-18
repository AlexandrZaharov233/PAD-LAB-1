from scada_rag.common import ROOT, digest, read_json


class Reranker:
    def __init__(self, name):
        from sentence_transformers import CrossEncoder
        revision = read_json(ROOT / "configs/model_revisions.json", {}).get(name)
        snapshot = ROOT / "models/reranker" / ("models--" + name.replace("/", "--")) / "snapshots" / (revision or "main")
        cached = any((snapshot / n).exists() for n in ("model.safetensors", "pytorch_model.bin")) and (snapshot / "tokenizer_config.json").exists()
        self.model = CrossEncoder(name, device="cpu", max_length=512,
                                  cache_dir=str(ROOT / "models/reranker"),
                                  revision=revision, local_files_only=cached)

    def rank(self, query, hits):
        import torch
        scores = self.model.predict([(query, x["text"]) for x in hits], batch_size=8,
                                    activation_fct=torch.nn.Sigmoid(), show_progress_bar=False)
        return sorted([{**x, "reranker_score": float(s)} for x, s in zip(hits, scores)],
                      key=lambda x: x["reranker_score"], reverse=True)


def filter_hits(hits, threshold, limit=None, reranker_threshold=None):
    result, seen = [], set()
    for hit in hits:
        key = digest(" ".join(hit["text"].casefold().split()))
        if hit["similarity"] < threshold or len(hit["text"]) < 40 or key in seen:
            continue
        if reranker_threshold is not None and hit.get("reranker_score", 0) < reranker_threshold:
            continue
        seen.add(key)
        result.append(hit)
        if limit is not None and len(result) >= limit:
            break
    return result
