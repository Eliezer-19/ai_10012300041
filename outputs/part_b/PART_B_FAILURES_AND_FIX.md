# PART B — Retrieval failures vs hybrid fix

**Setup**: Sentence Transformers embeddings + FAISS (inner product on normalized vectors = cosine similarity). **Extension**: **Hybrid search** — Reciprocal Rank Fusion (RRF) of dense top-N and TF-IDF top-N; queries with long digit runs (e.g. vote totals) additionally use a **numeric-aware** re-score.

## q1: Public Financial Management Act 2016 Act 921 Section 28 budget presentation...

- **Why vector-only can fail**: Dense retrieval often ranks broadly similar policy paragraphs first; the chunk that actually cites Act 921 for presenting the budget can sit lower in the ranking.
- **Fix**: Hybrid RRF: the TF‑IDF leg strongly matches rare tokens ('921', 'Section 28', 'Public Financial Management Act') and promotes the correct passage.

### Dense-only (top-k)

1. score=`0.5690` | budget_pdf | 2025 Expenditure Measures  251. Mr. Speaker, to contain public expenditure and bring public finances back on a  sustainable path, we will improve the quality and efficiency in public spending and  rationalise expenditure to spur growth.  252. Mr. Speaker, the following expendi...
2. score=`0.5305` | budget_pdf | 1. Right Honourable Speaker, on the authority of His Excellency the President, John  Dramani Mahama and pursuant to Articles 179 and 180 of the 1992 Constitution of the  Republic of Ghana, and sections 21 and 23 of the Public Financial Management Act,  2016 (Act 921), I respec...
3. score=`0.5255` | budget_pdf | [PAGE 74] Resetting the Economy for the Ghana We Want 2025 Budget  320. To further reduce risk on the debt portfolio, government will build sufficient cash  buffers to support effective implementation of the liability management strategies. This  will help smoothen the redempt...
4. score=`0.5240` | budget_pdf | [PAGE 16] Resetting the Economy for the Ghana We Want 2025 Budget  Management Act (2016) Act 921 to deploy a robust Public Financial Management (PFM)  System.  34. Even though Ghana’s Public Financial Management Act remains one of the best in the  world, its poor implementatio...
5. score=`0.5177` | budget_pdf | 72. Mr. Speaker, in addition, preliminary data compiled by the Ministry of Finance show  that MDAs have committed government through contracts awarded to the tune of over  GH¢194.3bn ( 16.5% of GDP) as shown in Table 10, most of which were without  commencement certificates. T...

### Hybrid (RRF + numeric when applicable) (top-k)

1. score=`0.1072` | budget_pdf | 1. Right Honourable Speaker, on the authority of His Excellency the President, John  Dramani Mahama and pursuant to Articles 179 and 180 of the 1992 Constitution of the  Republic of Ghana, and sections 21 and 23 of the Public Financial Management Act,  2016 (Act 921), I respec...
2. score=`0.1049` | budget_pdf | [PAGE 16] Resetting the Economy for the Ghana We Want 2025 Budget  Management Act (2016) Act 921 to deploy a robust Public Financial Management (PFM)  System.  34. Even though Ghana’s Public Financial Management Act remains one of the best in the  world, its poor implementatio...
3. score=`0.1049` | budget_pdf | [PAGE 74] Resetting the Economy for the Ghana We Want 2025 Budget  320. To further reduce risk on the debt portfolio, government will build sufficient cash  buffers to support effective implementation of the liability management strategies. This  will help smoothen the redempt...
4. score=`0.1000` | budget_pdf | [PAGE 1] THEME: Resetting The Economy For The Ghana We Want  [PAGE 3] The Budget Statement  and Economic Policy  of the Government of Ghana for the  2025 Financial Year  Presented to Parliament by  DR. CASSIEL ATO FORSON (MP)  MINISTER FOR FINANCE  On Tuesday March 11, 2025  O...
5. score=`0.0980` | budget_pdf | [PAGE 61] Resetting the Economy for the Ghana We Want 2025 Budget  Fiscal Policy Objectives  236. Mr. Speaker, consistent with Section 14 of the Public Financial Management Act, 2016  (Act 921), the fiscal policy objectives of this government is to support the economic  transf...

---

## q2: 946048 votes Greater Accra...

- **Why vector-only can fail**: Cosine similarity is not reliable for exact vote totals: semantically similar 'Greater Accra' election rows (other years/candidates) can outrank the row containing 946,048.
- **Fix**: After RRF, a numeric-aware stage boosts chunks whose text contains the query’s digit run (comma-insensitive), so the 2016 Mahama row surfaces.

### Dense-only (top-k)

1. score=`0.6073` | election_row | Ghana presidential election results: In 1992, in Greater Accra Region, Others (Others; code: OTHERS) received 48,619 votes (9.59%).
2. score=`0.5950` | election_row | Ghana presidential election results: In 2004, in Greater Accra Region, George Aggudey (CPP; code: OTHERS) received 12,600 votes (0.72%).
3. score=`0.5797` | election_row | Ghana presidential election results: In 2016, in Greater Accra Region, Bridget Dzogbenuku (PPP; code: OTHERS) received 12,922 votes (0.64%).
4. score=`0.5779` | election_row | Ghana presidential election results: In 1996, in Greater Accra Region, Edward Mahama (PNC; code: OTHERS) received 32,723 votes (2.68%).
5. score=`0.5728` | election_row | Ghana presidential election results: In 2016, in Greater Accra Region, Nana Konadu Agyeman Rawlings (NDP; code: OTHERS) received 1,028 votes (0.05%).

### Hybrid (RRF + numeric when applicable) (top-k)

1. score=`0.0638` | election_row | Ghana presidential election results: In 2016, in Greater Accra Region, John Dramani Mahama (NDC; code: NDC) received 946,048 votes (46.69%).
2. score=`0.0318` | election_row | Ghana presidential election results: In 1992, in Greater Accra Region, Others (Others; code: OTHERS) received 48,619 votes (9.59%).
3. score=`0.0308` | election_row | Ghana presidential election results: In 2008, in Greater Accra Region, Edward Mahama (PNC; code: OTHERS) received 6,174 votes (0.37%).
4. score=`0.0293` | election_row | Ghana presidential election results: In 2016, in Greater Accra Region, Ivor Kobina Greenstreet (CPP; code: OTHERS) received 2,061 votes (0.10%).
5. score=`0.0290` | election_row | Ghana presidential election results: In 2016, in Greater Accra Region, David Asibi Ayindenaba Apasera (PNC; code: OTHERS) received 690 votes (0.03%).

---

*Doc-type check*: dense top-1=`election_row` → hybrid top-1=`election_row`.

## q3: John Dramani Mahama votes percentage Greater Accra 2016...

- **Why vector-only can fail**: Many rows share the same candidate + year embeddingally; dense top-k can mix regions before the user’s intended row.
- **Fix**: Hybrid search rewards co-occurrence of '2016', 'Greater Accra', and the candidate name via the lexical leg.

### Dense-only (top-k)

1. score=`0.6658` | election_row | Ghana presidential election results: In 2016, in Greater Accra Region, John Dramani Mahama (NDC; code: NDC) received 946,048 votes (46.69%).
2. score=`0.6473` | election_row | Ghana presidential election results: In 2016, in Western North Region (old region: Western Region), John Dramani Mahama (NDC; code: NDC) received 169,900 votes (53.70%).
3. score=`0.6466` | election_row | Ghana presidential election results: In 2016, in Upper West Region, John Dramani Mahama (NDC; code: NDC) received 167,032 votes (58.37%).
4. score=`0.6452` | election_row | Ghana presidential election results: In 2016, in Upper East Region, John Dramani Mahama (NDC; code: NDC) received 271,796 votes (60.32%).
5. score=`0.6430` | election_row | Ghana presidential election results: In 2016, in Ahafo Region (old region: Brong Ahafo Region), John Dramani Mahama (NDC; code: NDC) received 98,272 votes (43.98%).

### Hybrid (RRF + numeric when applicable) (top-k)

1. score=`0.1080` | election_row | Ghana presidential election results: In 2016, in Greater Accra Region, John Dramani Mahama (NDC; code: NDC) received 946,048 votes (46.69%).
2. score=`0.1020` | election_row | Ghana presidential election results: In 2016, in Upper East Region, John Dramani Mahama (NDC; code: NDC) received 271,796 votes (60.32%).
3. score=`0.1013` | election_row | Ghana presidential election results: In 2016, in Western Region, John Dramani Mahama (NDC; code: NDC) received 285,938 votes (41.55%).
4. score=`0.0995` | election_row | Ghana presidential election results: In 2016, in Upper West Region, John Dramani Mahama (NDC; code: NDC) received 167,032 votes (58.37%).
5. score=`0.0971` | election_row | Ghana presidential election results: In 2016, in Northern Region, John Dramani Mahama (NDC; code: NDC) received 389,132 votes (56.08%).

---

*Doc-type check*: dense top-1=`election_row` → hybrid top-1=`election_row`.

