# Failure Cluster Analysis — Phase A

**Sinh viên:** Đinh Hồng Đăng  
**MSHV:** 2A202601480  
**Ngày:** 26/08/2026

## Aggregate RAGAS Scores

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.8604 | 0.6735 | 0.5651 |
| answer_relevancy | 0.7691 | 0.6772 | 0.8044 |
| context_precision | 0.4780 | 0.6760 | 0.4702 |
| context_recall | 0.8508 | 0.8114 | 0.8480 |
| **avg_score** | **0.7396** | **0.7095** | **0.6719** |

## Failure Cluster Matrix

| Worst metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 4 | 6 | 4 | 14 |
| answer_relevancy | 0 | 0 | 0 | 0 |
| context_precision | 16 | 10 | 6 | 32 |
| context_recall | 0 | 4 | 0 | 4 |

## Dominant Failure Analysis

- Dominant distribution: **factual** (20 worst-metric cases).
- Dominant metric: **context_precision** (32 cases).

The retriever generally finds relevant policy content, as shown by context recall above 0.81 in all distributions. However, it returns too many irrelevant or weakly related chunks, reducing context precision. This is most visible in direct factual questions where version and document metadata should narrow the candidate set. Adversarial questions also reduce faithfulness because version conflicts and negation traps are harder to resolve.

## Suggested Fixes

| Metric | Root cause | Suggested fix |
|---|---|---|
| context_precision | Too many irrelevant chunks | Improve reranking, source/version metadata filters, and reduce final context size |
| faithfulness | Version conflicts and unsupported inference | Add explicit current-version instructions and answer verification |
| context_recall | Multi-hop evidence split across chunks | Add parent-child retrieval and query decomposition |
| answer_relevancy | Query intent not fully captured | Improve prompt template and question-type routing |

The adversarial average (0.6719) is lower than factual (0.7396), which indicates the test set is exposing the expected version-conflict weaknesses.
