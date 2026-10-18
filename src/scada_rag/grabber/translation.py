"""Локальный перевод RO -> EN -> RU короткими предложениями, с кэшем."""
import os
import re
from scada_rag.common import ROOT, DATA, digest, read_json, write_json
from scada_rag.preprocessing.text import windows

MODELS = ["Helsinki-NLP/opus-mt-roa-en", "Helsinki-NLP/opus-mt-en-ru"]
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


def translate_blocks(blocks, log=print):
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    torch.set_num_threads(6)
    cache = str(ROOT / "models/translation-pivot")
    tasks, result = [], []
    for block in blocks:
        text = block["text"]
        if block["page"] == 1:
            text = re.split(r"Veaceslav\s+UNTILA", text, maxsplit=1)[0]
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        sentences = re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-ZĂÂÎȘȚ])", text)
        for sentence in sentences:
            for part in windows(sentence.strip(), 900, 0, boundary=True):
                if len(part) < 15:
                    continue
                path = DATA / "translation_cache" / (digest([MODELS, "sentences-fp32-v1", part]) + ".json")
                cached = read_json(path)
                row = {**block, "original_text": part, "translation_models": MODELS, "translated": True}
                if cached:
                    row["text"] = cached["text"]
                else:
                    tasks.append((len(result), part, path))
                result.append(row)
    if not tasks:
        return result
    revisions = read_json(ROOT / "configs/model_revisions.json", {})
    pairs = []
    log(f"Перевод ANRE: загрузка двух компактных моделей, {len(tasks)} предложений")
    for name in MODELS:
        tokenizer = MarianTokenizer.from_pretrained(name, cache_dir=cache, revision=revisions.get(name))
        model = MarianMTModel.from_pretrained(name, cache_dir=cache, revision=revisions.get(name)).eval()
        pairs.append((tokenizer, model))
    for start in range(0, len(tasks), 8):
        batch = tasks[start:start + 8]
        texts = [b[1] for b in batch]
        for tokenizer, model in pairs:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
            if inputs["input_ids"].shape[1] > 512:
                raise ValueError("Сегмент перевода превышает 512 токенов")
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=384, num_beams=2, no_repeat_ngram_size=4, do_sample=False)
            texts = tokenizer.batch_decode(output, skip_special_tokens=True)
        for (i, original, path), translated in zip(batch, texts):
            if len(re.findall(r"[А-Яа-яЁё]", translated)) < 5:
                translated = "Исходный фрагмент (таблица или обозначения): " + original
            result[i]["text"] = translated
            write_json(path, {"text": translated, "models": MODELS})
        log(f"Перевод ANRE: {min(start+8,len(tasks))}/{len(tasks)}")
    return result
