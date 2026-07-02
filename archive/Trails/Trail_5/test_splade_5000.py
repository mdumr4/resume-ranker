import os
import json
import torch
import pandas as pd
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import CrossEncoder
from time import time

print("="*60)
print("INITIALIZING SPLADE-ONLY NEURAL RETRIEVAL ENGINE (TOP 5000)")
print("="*60)

# --- 1. Load Data & Indices ---
print("Loading Tabular Metadata...")
df = pd.read_parquet("index/features_v2.parquet")

print("Loading SPLADE Sparse Matrix...")
splade_matrix = load_npz("index/splade_combined.npz")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Models onto {device}...")
# 1. SPLADE (Sparse Lexical)
HF_TOKEN = "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"
splade_tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3", token=HF_TOKEN)
splade_model = AutoModelForMaskedLM.from_pretrained("naver/splade-v3", use_safetensors=False, token=HF_TOKEN).to(device)
splade_model.eval()

# 2. Cross-Encoder (Deep Reasoning)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device=device)

# --- 2. The Sample JD (From LLM Parser) ---
jd_parsed = {
  "cross_encoder_chunks": {
    "career": "Backend Engineer responsible for building and scaling high-throughput data pipelines and APIs. Requires at least 3 years of industry experience working with large-scale data systems.",
    "skills": "Python, Java, Spark, Airflow, SQL, AWS",
    "profile": "Toronto, Canada. Willing to relocate. Open to work within 60 days. Expected salary up to 30 LPA.",
    "education": "B.E. or B.Tech in Computer Science or related field from a Tier 1 or Tier 2 university."
  },
  
  # Recruiter preferences for final penalty math
  "preferences": {
      "min_experience": 3,
      "max_notice_period": 60
  }
}

# --- 3. Retrieval Functions ---
def encode_splade(text):
    with torch.no_grad():
        inputs = splade_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = splade_model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.logits.size())
        logits = outputs.logits * mask
        vec = torch.max(torch.relu(logits), dim=1)[0].cpu().numpy()
        return csr_matrix(vec)

def rrf(lists, k=60):
    """Reciprocal Rank Fusion"""
    scores = {}
    for lst in lists:
        for rank, cid in enumerate(lst):
            if cid not in scores:
                scores[cid] = 0.0
            scores[cid] += 1.0 / (k + rank)
    
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_items]

# --- 4. EXECUTE PIPELINE ---
print("\nExecuting Search Pipeline (SPLADE Multi-Query)...")
start_time = time()

# A. Sparse Search (SPLADE Multi-Query)
q_career = encode_splade(jd_parsed["cross_encoder_chunks"]["career"])
q_skills = encode_splade(jd_parsed["cross_encoder_chunks"]["skills"])
q_profile = encode_splade(jd_parsed["cross_encoder_chunks"]["profile"])
q_edu = encode_splade(jd_parsed["cross_encoder_chunks"]["education"])

# Dot products against the massive combined matrix
score_career = splade_matrix.dot(q_career.T).toarray().flatten()
score_skills = splade_matrix.dot(q_skills.T).toarray().flatten()
score_profile = splade_matrix.dot(q_profile.T).toarray().flatten()
score_edu = splade_matrix.dot(q_edu.T).toarray().flatten()

# Get Top 10000 for each SPLADE chunk to ensure a massive pool for RRF
splade_career_cids = df.iloc[np.argsort(score_career)[::-1][:10000]]['candidate_id'].tolist()
splade_skills_cids = df.iloc[np.argsort(score_skills)[::-1][:10000]]['candidate_id'].tolist()
splade_profile_cids = df.iloc[np.argsort(score_profile)[::-1][:10000]]['candidate_id'].tolist()
splade_edu_cids = df.iloc[np.argsort(score_edu)[::-1][:10000]]['candidate_id'].tolist()

print(f"SPLADE Retrieval completed in {time() - start_time:.2f} seconds.")

# B. RRF Fusion (Sparse Streams Only)
print("Fusing Sparse streams via RRF into Top 5000...")
fusion_lists = [
    splade_career_cids, 
    splade_skills_cids, 
    splade_profile_cids, 
    splade_edu_cids
]
top_5000_cids = rrf(fusion_lists, k=60)[:5000]

# --- 5. CROSS-ENCODER SCORING ---
print(f"Executing 4-Pass Cross-Encoder on Top {len(top_5000_cids)} candidates...")
print("WARNING: This requires 20,000 forward passes! This will take ~10x longer.")
ce_start = time()

top_df = df[df['candidate_id'].isin(top_5000_cids)].copy()

# Prepare pairs
pairs_career = [[jd_parsed["cross_encoder_chunks"]["career"], row['text_career']] for _, row in top_df.iterrows()]
pairs_skills = [[jd_parsed["cross_encoder_chunks"]["skills"], row['text_skills']] for _, row in top_df.iterrows()]
pairs_profile = [[jd_parsed["cross_encoder_chunks"]["profile"], row['text_profile']] for _, row in top_df.iterrows()]
pairs_edu = [[jd_parsed["cross_encoder_chunks"]["education"], row['text_edu']] for _, row in top_df.iterrows()]

# Inference
top_df['score_ce_career'] = cross_encoder.predict(pairs_career)
top_df['score_ce_skills'] = cross_encoder.predict(pairs_skills)
top_df['score_ce_profile'] = cross_encoder.predict(pairs_profile)
top_df['score_ce_edu'] = cross_encoder.predict(pairs_edu)

# Sigmoid Normalization
def sigmoid(x): return 1 / (1 + np.exp(-x))
for col in ['score_ce_career', 'score_ce_skills', 'score_ce_profile', 'score_ce_edu']:
    top_df[col] = sigmoid(top_df[col])

# Combine CE Scores
# Weights: 40% Skills, 30% Career, 20% Profile, 10% Edu
top_df['ai_score'] = (
    (top_df['score_ce_skills'] * 0.40) +
    (top_df['score_ce_career'] * 0.30) +
    (top_df['score_ce_profile'] * 0.20) +
    (top_df['score_ce_edu'] * 0.10)
)

print(f"Cross-Encoder inference completed in {time() - ce_start:.2f} seconds.")

# --- 6. MATHEMATICAL PENALTIES ---
print("Applying Trust and Structural Penalties...")

# Penalty 1: Trust Violations
trust_penalty = (top_df['trust_date_violation'] | top_df['trust_salary_violation']).astype(float) * 0.15

# Penalty 2: Experience Deficit
exp_deficit = np.maximum(0, jd_parsed["preferences"]["min_experience"] - top_df['years_of_experience'])
exp_penalty = (exp_deficit > 0).astype(float) * 0.15

# Penalty 3: Notice Period
notice_penalty = (top_df['notice_period'] > jd_parsed["preferences"]["max_notice_period"]).astype(float) * 0.05

# Final Score Calculation
top_df['final_score'] = top_df['ai_score'] - trust_penalty - exp_penalty - notice_penalty

# Sort by final score
final_ranking = top_df.sort_values(by='final_score', ascending=False)

# Get Top 100 for Submission
top_100 = final_ranking.head(100).copy()

print("\n" + "="*60)
print(f"TOP 5 CANDIDATES FOR: Backend Engineer (From 5000 Pool)")
print("="*60)

for i, (_, row) in enumerate(top_100.head(5).iterrows()):
    print(f"\nRank {i+1} | Candidate: {row['candidate_id']} | Final Score: {row['final_score']:.4f}")
    print(f"  - AI Score: {row['ai_score']:.4f}")
    print(f"  - YOE: {row['years_of_experience']} | Notice: {row['notice_period']} days")
    if row['trust_date_violation']: print("  [!] WARNING: Trust Date Violation Detected")
    if row['trust_salary_violation']: print("  [!] WARNING: Trust Salary Violation Detected")

# --- 7. SAVE SUBMISSION ---
print("\nSaving Top 100 to submission.csv...")
job_id = "JD_1"
submission_records = []

for i, (_, row) in enumerate(top_100.iterrows()):
    submission_records.append({
        "job_id": job_id,
        "candidate_id": row["candidate_id"],
        "rank": i + 1,
        "score": row["final_score"]
    })

sub_df = pd.DataFrame(submission_records)
sub_out = "Trails/Trail_5/submission.csv"
sub_df.to_csv(sub_out, index=False)
print(f"Success! Saved {len(sub_df)} records to {sub_out}")
