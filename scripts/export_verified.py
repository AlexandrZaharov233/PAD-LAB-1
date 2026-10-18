"""Публикация результатов без повторяющихся полных текстов корпуса."""

import argparse
import json
import shutil
from pathlib import Path


def compact_hit(hit):
    return {key: hit[key] for key in ("id", "source_id", "page", "similarity", "reranker_score", "citation") if key in hit}


def export(source, target):
    source, target = Path(source), Path(target)
    if not source.is_dir() or not list(source.rglob("result.json")):
        raise ValueError("Не найдены результаты запуска")
    if target.exists():
        raise ValueError("Папка назначения уже существует: " + str(target))
    for file in sorted(source.rglob("*")):
        if not file.is_file() or file.name not in {"result.json", "comparison.json", "comparison.csv", "answers.jsonl", "manual_review.csv"}:
            continue
        dest = target / file.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if file.name != "answers.jsonl":
            shutil.copyfile(file, dest)
            continue
        with file.open(encoding="utf-8") as src, dest.open("w", encoding="utf-8") as out:
            for line in src:
                row = json.loads(line)
                for key in ("retrieved", "context"):
                    row[key] = [compact_hit(hit) for hit in row[key]]
                if "generation" in row:
                    row["generation"]["sources"] = [compact_hit(hit) for hit in row["generation"]["sources"]]
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Папка фактического запуска")
    parser.add_argument("target", help="Новая папка в experiments/verified")
    args = parser.parse_args()
    export(args.source, args.target)
