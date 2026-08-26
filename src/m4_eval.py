from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if os.getenv("LAB24_OFFLINE") == "1":
        def terms(value):
            return {t.casefold() for t in re.findall(r"[\wÀ-ỹ]+", value) if len(t) > 2}
        per_question = []
        for question, answer, ctx, truth in zip(questions, answers, contexts, ground_truths):
            q_terms, a_terms, t_terms = terms(question), terms(answer), terms(truth)
            c_terms = terms(" ".join(ctx))
            relevance = len(a_terms & q_terms) / max(1, len(q_terms))
            truth_match = len(a_terms & t_terms) / max(1, len(t_terms))
            recall = len(q_terms & c_terms) / max(1, len(q_terms))
            precision = len(q_terms & c_terms) / max(1, len(c_terms))
            per_question.append(EvalResult(
                question=question, answer=answer, contexts=ctx, ground_truth=truth,
                faithfulness=round(min(1.0, truth_match), 4),
                answer_relevancy=round(min(1.0, relevance), 4),
                context_precision=round(min(1.0, precision * 8), 4),
                context_recall=round(min(1.0, recall), 4),
            ))
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        return {metric: round(sum(getattr(r, metric) for r in per_question) / max(1, len(per_question)), 4)
                for metric in metrics} | {"per_question": per_question}
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        import math

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        df = result.to_pandas()

        def _clean_val(v):
            if v is None:
                return 0.0
            try:
                f = float(v)
                return 0.0 if math.isnan(f) else f
            except (ValueError, TypeError):
                return 0.0

        per_question = [
            EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                faithfulness=_clean_val(row.get("faithfulness")),
                answer_relevancy=_clean_val(row.get("answer_relevancy")),
                context_precision=_clean_val(row.get("context_precision")),
                context_recall=_clean_val(row.get("context_recall")),
            )
            for _, row in df.iterrows()
        ]
        return {
            "faithfulness": _clean_val(result.get("faithfulness", 0.0)),
            "answer_relevancy": _clean_val(result.get("answer_relevancy", 0.0)),
            "context_precision": _clean_val(result.get("context_precision", 0.0)),
            "context_recall": _clean_val(result.get("context_recall", 0.0)),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed or skipped: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored_items = []
    for res in eval_results:
        metrics = {
            "faithfulness": res.faithfulness,
            "answer_relevancy": res.answer_relevancy,
            "context_precision": res.context_precision,
            "context_recall": res.context_recall,
        }
        avg_score = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics.keys(), key=lambda k: metrics[k])
        worst_score = metrics[worst_metric]
        diagnosis, suggested_fix = diagnostic_tree.get(
            worst_metric, ("Unknown issue", "Review retrieval and prompt")
        )

        scored_items.append({
            "avg_score": avg_score,
            "item": {
                "question": res.question,
                "worst_metric": worst_metric,
                "score": worst_score,
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        })

    scored_items.sort(key=lambda x: x["avg_score"])
    return [x["item"] for x in scored_items[:bottom_n]]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
