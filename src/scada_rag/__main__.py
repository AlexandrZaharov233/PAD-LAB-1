import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="SCADA RAG - локальная лабораторная №1")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ["grab", "prepare", "index", "doctor"]:
        subs.add_parser(name)
    serve_parser = subs.add_parser("serve")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--open", action="store_true")
    ask = subs.add_parser("ask")
    ask.add_argument("question")
    evaluation = subs.add_parser("evaluate")
    evaluation.add_argument("--split", choices=["all", "dev", "test"], default="all")
    evaluation.add_argument("--generate", action="store_true")
    experiments = subs.add_parser("experiments")
    experiments.add_argument("--suite", choices=["retrieval", "generation"], default="retrieval")
    experiments.add_argument("--split", choices=["all", "dev", "test"], default="dev")
    experiments.add_argument("--limit", type=int, help="Необязательная воспроизводимая подвыборка, seed=42")
    args = parser.parse_args()
    try:
        if args.command == "serve":
            from scada_rag.server import serve
            return serve(args.port, args.open)
        if args.command == "doctor":
            from scada_rag.common import ROOT, config
            from scada_rag.grabber.collector import tesseract_path
            from scada_rag.generation.ollama import request_json
            print("Python:", sys.version)
            print("Проект:", ROOT)
            print("Tesseract:", tesseract_path() or "не найден")
            tags = request_json(config()["ollama_url"], "/api/tags", timeout=5)
            print("Модели Ollama:", [x["name"] for x in tags.get("models", [])])
            return
        if args.command in {"grab", "prepare"}:
            from scada_rag.grabber.collector import collect
            failures = [x for x in collect() if x["status"] == "error"]
            if failures:
                raise RuntimeError("Не все источники загружены: " + json.dumps(failures, ensure_ascii=False))
            if args.command == "grab":
                return
        if args.command == "experiments":
            from scada_rag.evaluation.runner import run_experiments
            print(run_experiments(args.suite, args.split, limit=args.limit))
            return
        from scada_rag.pipeline import Pipeline
        pipeline = Pipeline()
        if args.command == "prepare" and pipeline.cfg["use_reranker"]:
            pipeline.load_reranker()
        if args.command == "ask":
            print(json.dumps(pipeline.ask(args.question), ensure_ascii=False, indent=2))
        elif args.command == "evaluate":
            from scada_rag.evaluation.runner import evaluate, load_questions, save_run
            questions = load_questions(args.split)
            summary, rows = evaluate(pipeline, questions, generation=args.generate)
            folder = save_run("evaluation-" + args.split, pipeline.cfg, questions, summary, rows)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(folder)
        else:
            print("Индекс готов:", pipeline.store.meta["chunk_count"], "фрагментов")
    except Exception as exc:
        print("Ошибка:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
