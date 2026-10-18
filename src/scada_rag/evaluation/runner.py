import copy
import csv
import gc
import json
import platform
import random
import statistics
import time
from pathlib import Path

from scada_rag.common import ROOT, config, corpus_hash, documents, now, read_json, write_json
from scada_rag.evaluation.metrics import answer_metrics, retrieval_metrics, validate_evidence


def load_questions(split="all"):
    all_questions = read_json(ROOT / "data/questions.json")["questions"]
    return [q for q in all_questions if split == "all" or q["split"] == split]


def average(items):
    keys = {key for row in items for key in row}
    return {key: statistics.mean(values) if (values := [r[key] for r in items if r.get(key) is not None]) else None for key in keys}


def evaluate(pipeline, questions, cfg=None, generation=False, log=print):
    from scada_rag.generation.ollama import generate
    cfg = {**pipeline.cfg, **(cfg or {})}
    drift = validate_evidence(questions, pipeline.docs)
    if drift:
        raise ValueError("Эталонные фрагменты не найдены в текущем корпусе. Обновите разметку: " + json.dumps(drift, ensure_ascii=False))
    rows, raw_scores, context_scores, answer_scores, timings = [], [], [], [], []
    # Cache only query vectors for this exact model, not retrieval or reranking.
    vectors = pipeline.embedder.encode([q["question"] for q in questions], query=True)
    for i, q in enumerate(questions):
        result = pipeline.retrieve(q["question"], cfg, vectors[i:i + 1])
        raw = retrieval_metrics(result["retrieved"], q, cfg["top_k"])
        context = retrieval_metrics(result["context"], q, cfg["context_k"])
        if raw is not None:
            raw_scores.append(raw)
            context_scores.append(context)
        row = {"question_id": q["id"], "question": q["question"], "type": q["type"], "split": q["split"],
               "expected_answer": q["expected_answer"], "answerable": q["answerable"],
               "raw_metrics": raw, "context_metrics": context, **result}
        if generation:
            started = time.perf_counter()
            answer = generate(q["question"], result["context"], cfg)
            scores = answer_metrics(answer, q)
            answer_scores.append(scores)
            row.update({"generation": answer, "answer_metrics": scores, "generation_seconds": time.perf_counter() - started,
                        "faithfulness_manual": None, "answer_relevance_manual": None})
        timings.append(result["retrieval_seconds"])
        rows.append(row)
        log(f"Оценка {i+1}/{len(questions)}: {q['id']}")
    summary = {"question_count": len(questions), "answerable_count": len(raw_scores), "generation_enabled": generation,
               "raw": average(raw_scores), "context": average(context_scores), "answer": average(answer_scores),
               "retrieval_seconds_mean": statistics.mean(timings), "index": pipeline.store.meta,
               "timing_note": "Query embeddings вычислены пакетом до замера; первая загрузка reranker входит в время первого запроса."}
    return summary, rows


def save_run(name, cfg, questions, summary, rows, folder=None):
    folder = Path(folder) if folder else ROOT / "experiments/runs" / (time.strftime("%Y%m%d-%H%M%S") + "-" + name)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "result.json", {"name": name, "at": now(), "config": cfg, "summary": summary,
                                        "corpus_hash": corpus_hash(documents()), "python": platform.python_version(),
                                        "platform": platform.platform(), "questions": [q["id"] for q in questions]})
    with (folder / "answers.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    if summary["generation_enabled"]:
        with (folder / "manual_review.csv").open("w", encoding="utf-8-sig", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["question_id", "faithfulness_0_to_2", "relevance_0_to_2", "comment"])
            writer.writerows([[q["id"], "", "", ""] for q in questions])
    return folder


def run_experiments(suite="retrieval", split="dev", log=print, limit=None):
    from scada_rag.pipeline import Pipeline
    questions = load_questions(split)
    if limit:
        if limit < 2:
            raise ValueError("Для эксперимента нужны хотя бы два вопроса")
        questions = random.Random(42).sample(questions, min(limit, len(questions)))
    base = config()
    variants = [("baseline", {})]
    if suite == "retrieval":
        variants += [("fixed_600", {"chunk_strategy": "fixed", "chunk_size": 600, "chunk_overlap": 0}),
                     ("fixed_1000", {"chunk_strategy": "fixed", "chunk_overlap": 0}),
                     ("fixed_1400", {"chunk_strategy": "fixed", "chunk_size": 1400, "chunk_overlap": 0}),
                     ("overlap_0", {"chunk_strategy": "overlap", "chunk_overlap": 0}),
                     ("overlap_150", {"chunk_strategy": "overlap", "chunk_overlap": 150}),
                     ("overlap_300", {"chunk_strategy": "overlap", "chunk_overlap": 300}),
                     ("paragraphs", {"chunk_strategy": "paragraphs", "chunk_overlap": 0}),
                     ("k3", {"top_k": 3}), ("k5", {"top_k": 5}), ("k10", {"top_k": 10}),
                     ("without_reranker", {"use_reranker": False}),
                     ("threshold_068", {"similarity_threshold": 0.68}),
                     ("threshold_080", {"similarity_threshold": 0.80}),
                     ("e5_embedding_only", {"similarity_threshold": 0.0, "use_reranker": False}),
                     ("alternative_embedding_only", {"embedding_model": base["alternative_embedding_model"], "similarity_threshold": 0.0,
                                                      "use_reranker": False})]
    elif suite == "generation":
        variants += [("prompt_concise", {"prompt_variant": "concise"}), ("temperature_04", {"temperature": 0.4})]
    else:
        raise ValueError("Выберите retrieval или generation")
    folder = ROOT / "experiments/runs" / (time.strftime("%Y%m%d-%H%M%S") + "-suite-" + suite + "-" + split)
    summaries, pipeline, last_index = [], None, None
    for name, overrides in variants:
        cfg = config(overrides)
        identity = tuple(cfg[k] for k in ("embedding_model", "chunk_strategy", "chunk_size", "chunk_overlap"))
        if identity != last_index:
            if pipeline is not None and pipeline.cfg["embedding_model"] == cfg["embedding_model"]:
                from scada_rag.retrieval.store import VectorStore
                pipeline.cfg = cfg
                pipeline.store = VectorStore.build(cfg, pipeline.docs, pipeline.embedder, log)
            else:
                del pipeline
                gc.collect()
                pipeline = Pipeline(cfg, log)
            last_index = identity
        log("Эксперимент: " + name)
        summary, rows = evaluate(pipeline, questions, cfg, generation=suite == "generation", log=log)
        save_run(name, cfg, questions, summary, rows, folder / name)
        summaries.append({"name": name, "raw_recall": summary["raw"].get("recall"),
                          "context_recall": summary["context"].get("recall"), "context_mrr": summary["context"].get("mrr"),
                          "context_precision": summary["context"].get("precision"),
                          "seconds_mean": summary["retrieval_seconds_mean"], **summary["answer"]})
        write_json(folder / "comparison.json", summaries)
        with (folder / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            fields = list(dict.fromkeys(k for row in summaries for k in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summaries)
    return folder
