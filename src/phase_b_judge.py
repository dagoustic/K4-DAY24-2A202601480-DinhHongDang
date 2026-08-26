from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH, TEST_SET_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    # Implemented
    # PROMPT_TEMPLATE = '''Bạn là một expert đánh giá chất lượng câu trả lời RAG.
    #
    # Câu hỏi: {question}
    #
    # Answer A:
    # {answer_a}
    #
    # Answer B:
    # {answer_b}
    #
    # Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
    # Trả lời JSON (chỉ JSON, không text khác):
    # {{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
    # '''
    #
    # from openai import OpenAI
    # client = OpenAI()
    # resp = client.chat.completions.create(
    #     model=JUDGE_MODEL,
    #     messages=[
    #         {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
    #         {"role": "user",   "content": PROMPT_TEMPLATE.format(
    #             question=question, answer_a=answer_a, answer_b=answer_b)},
    #     ],
    #     response_format={"type": "json_object"},
    # )
    # return json.loads(resp.choices[0].message.content)
    prompt = f"""You are an expert RAG evaluator. Compare two answers for this HR-policy question.
Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}
Return JSON only with winner (A, B, or tie), reasoning, and scores A/B from 0 to 1.
"""
    if OPENAI_API_KEY and os.getenv("LAB24_OFFLINE") != "1":
        try:
            from openai import OpenAI
            response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": "Return valid JSON only."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            winner = data.get("winner", "tie")
            if winner not in {"A", "B", "tie"}:
                winner = "tie"
            scores = data.get("scores", {})
            return {"winner": winner, "reasoning": str(data.get("reasoning", "")),
                    "scores": {"A": max(0.0, min(1.0, float(scores.get("A", 0.0)))),
                               "B": max(0.0, min(1.0, float(scores.get("B", 0.0))))}}
        except Exception as exc:
            reasoning = f"LLM judge unavailable; deterministic fallback used ({type(exc).__name__})."
    else:
        reasoning = "No OPENAI_API_KEY; deterministic lexical baseline used."

    def tokens(text):
        return {token.lower().strip(".,!?;:()[]") for token in text.split() if len(token) > 2}
    q_tokens = tokens(question)
    def score(answer):
        overlap = len(tokens(answer) & q_tokens) / max(1, len(q_tokens))
        return min(1.0, 0.65 * overlap + 0.35 * min(len(answer) / 240.0, 1.0))
    scores = {"A": round(score(answer_a), 4), "B": round(score(answer_b), 4)}
    if abs(scores["A"] - scores["B"]) < 0.05:
        winner = "tie"
    else:
        winner = "A" if scores["A"] > scores["B"] else "B"
    return {"winner": winner, "reasoning": reasoning, "scores": scores}


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    # Implemented
    # pass1 = pairwise_judge(question, answer_a, answer_b)
    # pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!
    #
    # # Convert pass2 back to original A/B space
    # swap_map = {"A": "B", "B": "A", "tie": "tie"}
    # winner_pass2 = swap_map[pass2_raw["winner"]]
    #
    # # Average: consensus only if both agree
    # if pass1["winner"] == winner_pass2:
    #     final = pass1["winner"]
    # else:
    #     final = "tie"  # disagreement = inconclusive
    #
    # position_consistent = (pass1["winner"] == winner_pass2)
    #
    # return JudgeResult(
    #     question=question, answer_a=answer_a, answer_b=answer_b,
    #     winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
    #     final_winner=final,
    #     reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
    #     position_consistent=position_consistent,
    #     scores_pass1=pass1["scores"],
    #     scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    # )
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)
    winner_pass2 = {"A": "B", "B": "A", "tie": "tie"}[pass2_raw["winner"]]
    consistent = pass1["winner"] == winner_pass2
    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=pass1["winner"] if consistent else "tie",
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=consistent,
        scores_pass1=pass1.get("scores", {}),
        scores_pass2={"A": pass2_raw.get("scores", {}).get("B", 0.0),
                      "B": pass2_raw.get("scores", {}).get("A", 0.0)},
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    if len(judge_labels) != len(human_labels) or not judge_labels:
        raise ValueError("judge_labels and human_labels must have equal non-zero length")
    n = len(judge_labels)
    observed = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    judge_pos = sum(j == 1 for j in judge_labels) / n
    human_pos = sum(h == 1 for h in human_labels) / n
    expected = judge_pos * human_pos + (1 - judge_pos) * (1 - human_pos)
    return round((observed - expected) / (1 - expected), 6) if expected != 1 else 1.0


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    # total = len(judge_results)
    # if total == 0:
    #     return {"total_judged": 0, "position_bias_rate": 0.0, "verbosity_bias": 0.0}
    #
    # position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    # position_bias_rate  = position_bias_count / total
    #
    # a_wins_a_longer = sum(
    #     1 for r in judge_results
    #     if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    # )
    # b_wins_b_longer = sum(
    #     1 for r in judge_results
    #     if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    # )
    # decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    # verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0
    #
    # interpretation = ("Position bias cao — nên dùng swap-and-average."
    #                   if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định.")
    # return {
    #     "total_judged": total, "position_bias_rate": round(position_bias_rate, 3),
    #     "position_bias_count": position_bias_count,
    #     "verbosity_bias": round(verbosity_bias, 3),
    #     "verbosity_details": {"a_wins_a_longer": a_wins_a_longer,
    #                           "b_wins_b_longer": b_wins_b_longer,
    #                           "total_decisive": decisive},
    #     "interpretation": interpretation,
    # }
    total = len(judge_results)
    inconsistent = sum(not r.position_consistent for r in judge_results)
    decisive = [r for r in judge_results if r.final_winner != "tie"]
    a_longer = sum(r.final_winner == "A" and len(r.answer_a) > len(r.answer_b) for r in decisive)
    b_longer = sum(r.final_winner == "B" and len(r.answer_b) > len(r.answer_a) for r in decisive)
    verbosity = (a_longer + b_longer) / len(decisive) if decisive else 0.0
    position_rate = inconsistent / total if total else 0.0
    return {
        "total_judged": total,
        "position_bias_rate": round(position_rate, 3),
        "position_bias_count": inconsistent,
        "verbosity_bias": round(verbosity, 3),
        "verbosity_details": {"a_wins_a_longer": a_longer,
                               "b_wins_b_longer": b_longer,
                               "total_decisive": len(decisive)},
        "interpretation": ("Position bias cao — nên dùng swap-and-average."
                            if position_rate > 0.3 else
                            "Position bias thấp — judge tương đối ổn định."),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_phase_b_report() -> dict:
    """Run the ten human-labelled comparisons and persist a reproducible report."""
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_by_id = {item["id"]: item for item in json.load(f)}
    results = []
    judge_labels = []
    for item in human_data:
        reference = test_by_id.get(item["question_id"], {}).get("ground_truth", item.get("model_answer", ""))
        judged = swap_and_average(item["question"], item["model_answer"], reference)
        results.append(judged)
        judge_labels.append(1 if judged.final_winner in {"A", "tie"} else 0)
    human_labels = [item["human_label"] for item in human_data]
    kappa = cohen_kappa(judge_labels, human_labels)
    bias = bias_report(results)
    report = {
        "student": "Đinh Hồng Đăng", "student_id": "2A202601480", "date": "26/08/2026",
        "judge_model": JUDGE_MODEL, "num_questions": len(results),
        "judge_labels": judge_labels, "human_labels": human_labels,
        "cohen_kappa": kappa, "bias": bias,
        "results": [{"question": r.question, "winner_pass1": r.winner_pass1,
                      "winner_pass2": r.winner_pass2, "final_winner": r.final_winner,
                      "position_consistent": r.position_consistent,
                      "reasoning_pass1": r.reasoning_pass1,
                      "reasoning_pass2": r.reasoning_pass2}
                     for r in results],
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    # --- Demo pairwise + swap ---
    q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
    a_a = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành."
    a_b = "Theo quy định, nhân viên có 12 ngày phép hàng năm."

    print("Running swap-and-average judge...")
    result = swap_and_average(q, a_a, a_b)
    print(f"  Pass 1 winner: {result.winner_pass1}")
    print(f"  Pass 2 winner: {result.winner_pass2}")
    print(f"  Final:         {result.final_winner}")
    print(f"  Position consistent: {result.position_consistent}")

    # --- Cohen's κ vs human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nHuman labels loaded: {len(human_labels)} questions")

    # In production: run judge on the same 10 questions to get judge_labels
    judge_labels = [0] * len(human_labels)  # Demo labels; production reports use actual judge outputs.
    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"Cohen's κ: {kappa:.3f}")

    # --- Full Phase B report ---
    report = run_phase_b_report()
    print(f"\nCohen's κ: {report['cohen_kappa']:.3f}")
    print(f"Bias report: {report['bias']}")
