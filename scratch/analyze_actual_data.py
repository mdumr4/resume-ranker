import json
import csv

def main():
    # Read the 500 candidates
    candidates = {}
    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 500: break
            c = json.loads(line)
            candidates[c['candidate_id']] = c
            
    # Read ranks
    ranks = {}
    with open("Trails/Trail_1/submission.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranks[row['candidate_id']] = int(row['rank'])
            
    # Keywords indicating real experience in ranking/search/recsys
    career_keywords = ["ranking", "recommendation", "search", "xgboost", "retrieval"]
    
    missed_candidates = []
    
    for c_id, c in candidates.items():
        rank = ranks.get(c_id, 999)
        
        # Check career history for real experience
        history = c.get('career_history', [])
        career_text = " ".join([h.get('description', '').lower() for h in history])
        
        has_real_exp = any(k in career_text for k in career_keywords)
        
        # Check skills for buzzwords
        skills = [s.get('name', '').lower() for s in c.get('skills', [])]
        has_buzzwords = any("faiss" in s or "data analyst" in s for s in skills)
        
        # If they have real experience but ranked poorly (e.g. outside top 100)
        if has_real_exp and rank > 100:
            missed_candidates.append({
                "id": c_id,
                "name": c.get('profile', {}).get('anonymized_name'),
                "rank": rank,
                "career_snippet": career_text[:300] + "...",
                "skills": skills
            })
            
    print(f"Found {len(missed_candidates)} candidates with actual career experience in search/ranking who ranked > 100.")
    for m in missed_candidates[:5]:
        print(f"\n[{m['rank']}] {m['name']} ({m['id']})")
        print(f"Career: {m['career_snippet']}")
        print(f"Skills: {m['skills']}")

if __name__ == "__main__":
    main()
