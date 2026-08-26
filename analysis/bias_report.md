# LLM Judge Bias Report — Phase B

**Sinh viên:** Đinh Hồng Đăng  
**MSHV:** 2A202601480  
**Ngày:** 26/08/2026  
**Judge model:** gpt-4o-mini (offline deterministic fallback used for this run)

## Summary

| Metric | Result |
|---|---:|
| Questions judged | 10 |
| Cohen's kappa | -0.1538 |
| Position bias rate | 0.0% |
| Verbosity bias | 85.7% |

## Interpretation

Swap-and-average produced consistent decisions for all ten comparisons, so no position inconsistency was observed in this run. However, the negative Cohen's kappa means the deterministic fallback judge disagreed with human labels more than expected by chance; it must not be used as a production evaluator. The high verbosity-bias rate suggests the scoring heuristic tends to prefer the longer reference answer. With a live LLM judge, the same ten cases should be rerun and this report updated before using the result as a CI quality gate.

Detailed machine-readable results are in `reports/judge_results.json`.
