import os
from scada_rag.common import ROOT, read_json

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


class Embedder:
    def __init__(self, name, threads=6):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(threads)
        self.name = name
        self.revision = read_json(ROOT / "configs/model_revisions.json", {}).get(name)
        cache = ROOT / "models/embeddings"
        snapshot = cache / ("models--" + name.replace("/", "--")) / "snapshots" / (self.revision or "main")
        cached = (snapshot / "modules.json").exists() and any((snapshot / n).exists() for n in ("model.safetensors", "pytorch_model.bin"))
        self.model = SentenceTransformer(name, device="cpu", cache_folder=str(cache), revision=self.revision, local_files_only=cached)
        self.model.max_seq_length = 512

    def encode(self, texts, query=False):
        if "e5" in self.name.lower():
            prefix = "query: " if query else "passage: "
            texts = [prefix + text for text in texts]
        return self.model.encode(texts, normalize_embeddings=True, batch_size=16, convert_to_numpy=True,
                                 show_progress_bar=False).astype("float32")

    def truncation_count(self, texts):
        return sum(len(self.model.tokenizer.encode(t, add_special_tokens=True)) > 512 for t in texts)
