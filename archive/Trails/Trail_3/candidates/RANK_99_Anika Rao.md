# Candidate: Anika Rao
**Candidate ID**: CAND_0055905
**Headline**: Senior Machine Learning Engineer | LLMs, RAG, Vector Search | ex-Top Tech
**Location**: London, UK | **Experience**: 8.1 Years

## Summary
Senior AI engineer with 8.1 years of hands-on experience building production ML systems, with a focus on search, retrieval, and ranking. Most recently, I led the migration from keyword-based ranking to a learning-to-rank model with embedded behavioral signals, serving 50M+ queries per month. My day-to-day work spans embedding model selection and fine-tuning, hybrid retrieval architecture, learning-to-rank, behavioral-signal integration, and the offline/online evaluation that ties it all together. I've shipped systems in both early-stage product companies and at larger scale, and I've spent enough time on both that I know which tradeoffs apply where. I care more about shipping a working system in 6 weeks than a theoretically perfect one in 6 months. Currently exploring my next move — looking for senior IC or tech-lead roles where I can own the intelligence layer end-to-end.

## Skills
- **Elasticsearch** (expert) — 23 endorsement(s), 80 month(s)
- **ASR** (advanced) — 56 endorsement(s), 41 month(s)
- **Hugging Face Transformers** (advanced) — 54 endorsement(s), 33 month(s)
- **Haystack** (advanced) — 6 endorsement(s), 26 month(s)
- **Speech Recognition** (advanced) — 20 endorsement(s), 57 month(s)
- **LangChain** (expert) — 33 endorsement(s), 70 month(s)
- **Python** (expert) — 34 endorsement(s), 45 month(s)
- **LLMs** (advanced) — 50 endorsement(s), 43 month(s)
- **OpenSearch** (expert) — 57 endorsement(s), 56 month(s)
- **Fine-tuning LLMs** (advanced) — 57 endorsement(s), 50 month(s)
- **Information Retrieval** (advanced) — 8 endorsement(s), 60 month(s)
- **Embeddings** (expert) — 17 endorsement(s), 75 month(s)
- **Vector Search** (expert) — 1 endorsement(s), 93 month(s)

## Career History
### Senior Machine Learning Engineer at Flipkart
**Duration**: 2025-04-02 to Present (14 months) | **Current**: True
Owned the design and rollout of a large-scale semantic search system serving an internal corpus of 35M+ items. Migrated the existing BM25-only retrieval to a hybrid setup combining sparse and dense vectors (sentence-transformers, MPNet-base initially, later fine-tuned BGE-large for our domain). The new system reduced p95 retrieval latency by 60% while improving NDCG@10 by 18% on our held-out eval set. Spent substantial time on the boring-but-critical parts: incremental index refresh, embedding drift monitoring, online/offline metric correlation. Led a team of 4 engineers across the rollout.

### Senior AI Engineer at Uber
**Duration**: 2022-03-19 to 2025-04-02 (37 months) | **Current**: False
Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA and QLoRA for domain-specific candidate-JD matching. Built the data curation pipeline that generated 200K high-quality preference pairs from recruiter labels, plus the eval harness using both ranking metrics and human-quality scores. Deployed the model via BentoML on Kubernetes with sub-200ms p95 latency by quantizing to INT8 and batching at the request level. Cost per inference dropped from $0.04 with GPT-3.5-fallback to under $0.001.

### Senior Applied Scientist at Rephrase.ai
**Duration**: 2018-05-09 to 2022-01-18 (45 months) | **Current**: False
Built a RAG-based ranking pipeline serving 50M+ queries per month for an internal recruiter-facing search product. The architecture combined BM25 + dense retrieval (BGE embeddings, FAISS HNSW) with an LLM-based re-ranker on the top-50, falling back to a learning-to-rank model when latency budget was tight. Designed the offline evaluation framework from scratch — NDCG, MRR, recall@K calibrated against online A/B engagement metrics. Drove the migration over 4 months including the recruiter-feedback loop that surfaced reranking edge cases.

## Education
- **B.Tech** in *Computer Engineering*, Anna University (2009 - 2013)
  - Grade: 7.12 CGPA | Tier: tier_2
- **M.S.** in *Data Science*, IIT Kharagpur (2002 - 2005)
  - Grade: 9.14 CGPA | Tier: tier_1

## Certifications
No certifications listed.

## Languages
- **English** (professional)
- **Hindi** (professional)

## Recruiter & Platform Signals
- **Profile Completeness**: 56.7%
- **Open to Work**: True
- **Notice Period**: 30 days
- **Expected Salary Range**: 38.9 - 64.1 INR LPA
- **Preferred Work Mode**: remote
- **Willing to Relocate**: False
- **GitHub Activity Score**: -1
- **Interview Completion Rate**: 0.67
- **Offer Acceptance Rate**: 0.93
