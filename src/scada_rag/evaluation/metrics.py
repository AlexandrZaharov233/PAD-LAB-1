import re
import unicodedata


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")).strip()


def relevant(hit, evidence):
    return hit["source_id"] == evidence["source_id"] and all(norm(x) in norm(hit["text"]) for x in evidence["contains"])


def retrieval_metrics(hits, question, k):
    evidence = question.get("evidence", [])
    if not evidence:
        return None
    hits = hits[:k]
    flags = [any(relevant(hit, e) for e in evidence) for hit in hits]
    covered = sum(any(relevant(hit, e) for hit in hits) for e in evidence)
    return {"recall": covered / len(evidence), "precision": sum(flags) / k,
            "hit_rate": float(any(flags)), "mrr": next((1 / (i + 1) for i, x in enumerate(flags) if x), 0.0),
            "all_evidence": float(covered == len(evidence))}


def validate_evidence(questions, docs):
    errors = []
    for question in questions:
        for evidence in question.get("evidence", []):
            text = " ".join(b["text"] for d in docs if d["id"] == evidence["source_id"] for b in d["blocks"])
            if not text or not all(norm(x) in norm(text) for x in evidence["contains"]):
                errors.append({"question": question["id"], "evidence": evidence})
    return errors


def answer_metrics(answer, question):
    expected = question["answerable"]
    result = {"refusal_correct": float(answer["answerable"] == expected)}
    if expected:
        phrases = question.get("answer_keywords", [])
        result["keyword_coverage"] = (sum(norm(p) in norm(answer["answer"]) for p in phrases) / len(phrases)) if phrases else None
    else:
        result["false_answer"] = float(answer["answerable"])
    # Valid citations only establish membership in context, not factual support.
    result["citation_membership"] = float(answer["validation"] == "citations_in_context") if answer["answerable"] else None
    return result
