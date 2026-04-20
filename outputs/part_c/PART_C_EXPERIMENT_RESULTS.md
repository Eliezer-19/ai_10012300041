# PART C — Prompt experiments (2026-04-20T11:33:00)

- **Generator**: OllamaChatGenerator (llama3 @ http://127.0.0.1:11434)
- **Retrieval**: hybrid RRF from `outputs/part_b`, top-8, packed ≤4500 chars.

## exp1: What is the stated theme of the 2025 Budget Statement of Ghana?

- **Context blocks used**: 8 (2778 chars)

### v1_minimal — Minimal instruction

*Context + question only; no explicit abstention or citation rules.*

**Metrics (heuristic)**:
- words=19, citation_markers=0, abstain_signal=False
- contains_ground_truth_keyword: **True** (theme wording)

**Model output**:

```
The stated theme of the 2025 Budget Statement of Ghana is "Resetting The Economy For The Ghana We Want".
```

### v2_grounded — Grounded + abstain + cite

*Use only passages; abstain if unsupported; cite [n] for facts.*

**Metrics (heuristic)**:
- words=25, citation_markers=2, abstain_signal=False
- contains_ground_truth_keyword: **True** (theme wording)

**Model output**:

```
According to passage [4], the theme of the 2025 Budget Statement of Ghana is:

"Resetting The Economy For The Ghana We Want"

[4] [PAGE 1-3]
```

### v3_json — Structured JSON

*Same grounding as v2; output JSON for auditable abstain/citations.*

**Metrics (heuristic)**:
- words=16, citation_markers=1, abstain_signal=False
- contains_ground_truth_keyword: **True** (theme wording)

**Model output**:

```
{
"answer": "THEME: Resetting The Economy For The Ghana We Want",
"citations": [4],
"abstain": false
}
```

---

## exp2: What is the capital city of France?

- **Context blocks used**: 3 (3301 chars)

### v1_minimal — Minimal instruction

*Context + question only; no explicit abstention or citation rules.*

**Metrics (heuristic)**:
- words=79, citation_markers=0, abstain_signal=False

**Model output**:

```
I'm happy to help you with your question! However, I don't see any information about France or its capital in the provided documents. The documents appear to be related to the 2025 budget statement and economic policy for Ghana, specifically focusing on the Northern Region and Savannah Region.

If you meant to ask a different question or provide more context, please feel free to do so! I'm here to assist you with any questions or concerns you may have.
```

### v2_grounded — Grounded + abstain + cite

*Use only passages; abstain if unsupported; cite [n] for facts.*

**Metrics (heuristic)**:
- words=6, citation_markers=0, abstain_signal=True

**Model output**:

```
Not found in the provided context.
```

### v3_json — Structured JSON

*Same grounding as v2; output JSON for auditable abstain/citations.*

**Metrics (heuristic)**:
- words=12, citation_markers=0, abstain_signal=True

**Model output**:

```
{
"answer": "Not enough information to answer",
"citations": [],
"abstain": true
}
```

---

## Analysis — observed differences (see also `DOCUMENTATION.md` in project root)

- **v1_minimal**: minimal guardrails; on the OOD geography probe the model may answer from **prior knowledge** (e.g. “Paris”) even when retrieved passages are unrelated.
- **v2_grounded**: explicit **abstention** phrase and **context-only** rules often fix this: the model outputs the required **Not found in the provided context.** when passages do not support the answer.
- **v3_json**: JSON is useful for UI/validators, but smaller Ollama models may still **hallucinate structured fields** (e.g. `abstain: false` with invented citations). Treat JSON as **best-effort** unless you add schema validation or use a larger / stronger Ollama model.
