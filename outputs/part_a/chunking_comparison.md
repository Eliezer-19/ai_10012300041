## PART A — Chunking comparison (generated 2026-04-20T11:31:23)
### Summary metrics
| Strategy | #chunks | Avg chars/chunk | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| A: fixed_chars (1200, overlap 200) | 1311 | 704 | 1.000 | 0.875 |
| B: fixed_words (260, overlap 50) | 1078 | 867 | 1.000 | 0.875 |
| C: paragraph_packed (max 1400, overlap 1 para) | 1130 | 725 | 1.000 | 0.792 |

### Notes
- Retrieval uses the **same TF‑IDF + cosine similarity** pipeline for all strategies.
- Queries are mixed across the PDF and CSV to reflect the combined dataset.
- Ground truth uses keyword inclusion to keep the comparison consistent and reproducible.

