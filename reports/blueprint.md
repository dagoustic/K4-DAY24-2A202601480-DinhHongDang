# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Đinh Hồng Đăng  
**MSHV:** 2A202601480  
**Ngày:** 26/08/2026

## Guard Stack Pipeline

| Layer | Tool | Latency P95 | Failure Action |
|---|---|---:|---|
| PII Detection | Presidio + VN regex recognizers | 0.03 ms | Reject + log |
| Topic/Jailbreak | NeMo Input Rail / offline lexical fallback | 0.57 ms | Reject + reason |
| RAG Pipeline | Day 18 | measured separately | Fallback |
| Output Check | NeMo Output Rail | n/a | Block + log |

## CI Gates

- RAGAS faithfulness >= 0.75.
- Adversarial suite pass rate >= 90% (18/20).
- P95 total guard latency < 500 ms.
- `pytest tests/ -q` passes.

## Monitoring Results

| Metric | Result |
|---|---:|
| RAGAS avg_score (50q) | 0.7140 |
| RAGAS faithfulness | 0.7266 |
| Worst RAGAS metric | context_precision (0.5556) |
| Dominant failure distribution | factual |
| Cohen's kappa | -0.1538 |
| Adversarial pass rate | 20/20 (100%) |
| Guard P95 latency | 0.60 ms |

## Interpretation

Phase A cho thấy context precision là điểm yếu chính, đặc biệt ở factual và adversarial questions; cần cải thiện reranking và metadata filtering. Adversarial guardrail đạt 20/20 nhờ kết hợp nhận diện PII với input keyword rail. Cohen's kappa âm cho thấy deterministic judge hiện chưa đủ tin cậy để thay thế human evaluation. Các số đo latency là offline baseline; khi bật NeMo gọi LLM thật cần đo lại và cập nhật CI gate.

## Suggested CI Commands

```powershell
$env:LAB24_OFFLINE = "1"
python src/phase_a_ragas.py
python src/phase_b_judge.py
python src/phase_c_guard.py
pytest tests/ -q
python check_lab.py
```
