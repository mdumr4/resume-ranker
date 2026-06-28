# Candidate: Arjun Khanna
**Candidate ID**: CAND_0081846
**Headline**: Lead AI Engineer | LLMs, RAG, Vector Search | ex-Top Tech
**Location**: Jaipur, Rajasthan, India | **Experience**: 6.7 Years

## Summary
Senior AI engineer with 6.7 years of hands-on experience building production ML systems, with a focus on search, retrieval, and ranking. Most recently, I rebuilt the candidate-JD matching pipeline from scratch, taking it from 0.72 to 0.91 NDCG@10, serving 50M+ queries per month. My day-to-day work spans embedding model selection and fine-tuning, hybrid retrieval architecture, learning-to-rank, behavioral-signal integration, and the offline/online evaluation that ties it all together. I've shipped systems in both early-stage product companies and at larger scale, and I've spent enough time on both that I know which tradeoffs apply where. I have strong opinions about when LLMs are the right hammer and when classical IR is — usually it's both. Currently exploring my next move — looking for senior IC or tech-lead roles where I can own the intelligence layer end-to-end.

## Skills
- **Data Science** (advanced) — 30 endorsement(s), 19 month(s)
- **Information Retrieval** (expert) — 59 endorsement(s), 93 month(s)
- **LlamaIndex** (advanced) — 5 endorsement(s), 39 month(s)
- **pgvector** (advanced) — 11 endorsement(s), 38 month(s)
- **Forecasting** (advanced) — 26 endorsement(s), 51 month(s)
- **Learning to Rank** (expert) — 54 endorsement(s), 58 month(s)
- **Elasticsearch** (expert) — 23 endorsement(s), 60 month(s)
- **PyTorch** (advanced) — 37 endorsement(s), 27 month(s)
- **Vector Search** (expert) — 53 endorsement(s), 79 month(s)
- **scikit-learn** (advanced) — 3 endorsement(s), 20 month(s)
- **Deep Learning** (advanced) — 2 endorsement(s), 40 month(s)
- **Recommendation Systems** (expert) — 36 endorsement(s), 62 month(s)
- **Python** (expert) — 16 endorsement(s), 75 month(s)
- **Embeddings** (expert) — 4 endorsement(s), 95 month(s)
- **Semantic Search** (advanced) — 8 endorsement(s), 19 month(s)
- **BM25** (expert) — 40 endorsement(s), 75 month(s)
- **Machine Learning** (expert) — 53 endorsement(s), 68 month(s)
- **Qdrant** (expert) — 38 endorsement(s), 49 month(s)

## Career History
### Lead AI Engineer at Razorpay
**Duration**: 2024-03-08 to Present (27 months) | **Current**: True
Built a RAG-based ranking pipeline serving 50M+ queries per month for an internal recruiter-facing search product. The architecture combined BM25 + dense retrieval (BGE embeddings, FAISS HNSW) with an LLM-based re-ranker on the top-50, falling back to a learning-to-rank model when latency budget was tight. Designed the offline evaluation framework from scratch — NDCG, MRR, recall@K calibrated against online A/B engagement metrics. Drove the migration over 4 months including the recruiter-feedback loop that surfaced reranking edge cases.

### Senior Machine Learning Engineer at Paytm
**Duration**: 2019-11-30 to 2024-03-08 (52 months) | **Current**: False
Owned the design and rollout of a large-scale semantic search system serving an internal corpus of 35M+ items. Migrated the existing BM25-only retrieval to a hybrid setup combining sparse and dense vectors (sentence-transformers, MPNet-base initially, later fine-tuned BGE-large for our domain). The new system reduced p95 retrieval latency by 60% while improving NDCG@10 by 18% on our held-out eval set. Spent substantial time on the boring-but-critical parts: incremental index refresh, embedding drift monitoring, online/offline metric correlation. Led a team of 4 engineers across the rollout.

## Education
- **B.E.** in *Data Science*, IIT Hyderabad (2006 - 2009)
  - Grade: 83% | Tier: tier_1
- **Ph.D** in *Computer Engineering*, IIT Delhi (2015 - 2019)
  - Grade: 8.39 CGPA | Tier: tier_1

## Certifications
- **Google Cloud Professional ML Engineer** issued by Google Cloud (2021)

## Languages
- **English** (professional)
- **Hindi** (conversational)

## Recruiter & Platform Signals
- **Profile Completeness**: 95.7%
- **Open to Work**: True
- **Notice Period**: 30 days
- **Expected Salary Range**: 33.3 - 52.8 INR LPA
- **Preferred Work Mode**: remote
- **Willing to Relocate**: True
- **GitHub Activity Score**: 33.7
- **Interview Completion Rate**: 0.94
- **Offer Acceptance Rate**: -1
