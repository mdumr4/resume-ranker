import pandas as pd
import json
import os
import argparse

def generate_shift_mds(shift_csv, jsonl_path="candidates.jsonl", out_dir="shift_candidates"):
    print(f"Loading {shift_csv}...")
    try:
        df = pd.read_csv(shift_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {shift_csv}")
        return

    # Sort by rank_shift
    # Positive rank_shift = Winner (Went up in Trail B)
    # Negative rank_shift = Loser (Went down in Trail B)
    winners = df.sort_values('rank_shift', ascending=False).head(10)
    losers = df.sort_values('rank_shift', ascending=True).head(10)
    
    target_ids = {}
    for i, row in winners.iterrows():
        shift = int(row['rank_shift'])
        target_ids[row['candidate_id']] = f"Winner_Up_{shift}_spots"
        
    for i, row in losers.iterrows():
        shift = int(row['rank_shift'])
        target_ids[row['candidate_id']] = f"Loser_Down_{abs(shift)}_spots"
        
    print(f"Searching for {len(target_ids)} candidates in {jsonl_path}...")
    
    os.makedirs(out_dir, exist_ok=True)
    
    found_count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            cand = json.loads(line)
            cid = cand.get('candidate_id')
            
            if cid in target_ids:
                rank_label = target_ids[cid]
                filepath = os.path.join(out_dir, f"{rank_label}_{cid}.md")
                
                # Build beautiful Markdown
                md_content = f"# Candidate: {cid} ({rank_label})\n\n"
                
                # Redrob Signals
                sig = cand.get('redrob_signals', {})
                md_content += "## Redrob Signals (Trust Data)\n"
                md_content += f"- **Profile Completeness**: {sig.get('profile_completeness_score', 'N/A')}%\n"
                md_content += f"- **Avg Response Time**: {sig.get('avg_response_time_hours', 'N/A')} hours\n"
                md_content += f"- **Open to Work**: {sig.get('open_to_work_flag', False)}\n"
                md_content += f"- **Verified Email/Phone**: {sig.get('verified_email', False)} / {sig.get('verified_phone', False)}\n\n"
                
                # Profile
                prof = cand.get('profile', {})
                md_content += "## Profile & Logistics\n"
                md_content += f"- **Location**: {prof.get('location', 'N/A')}, {prof.get('country', 'N/A')}\n"
                md_content += f"- **Total YOE**: {prof.get('years_of_experience', 'N/A')}\n"
                
                sal = sig.get('expected_salary_range_inr_lpa', {})
                md_content += f"- **Expected Salary**: {sal.get('min', 0)} - {sal.get('max', 0)} LPA\n"
                md_content += f"- **Notice Period**: {sig.get('notice_period_days', 'N/A')} days\n"
                md_content += f"- **Work Mode**: {sig.get('preferred_work_mode', 'N/A')}\n"
                md_content += f"- **Willing to Relocate**: {sig.get('willing_to_relocate', False)}\n"
                
                # Career
                md_content += "## Career History\n"
                for job in cand.get('career_history', []):
                    md_content += f"### {job.get('title', 'N/A')} at {job.get('company', 'N/A')}\n"
                    md_content += f"- **Duration**: {job.get('duration_months', 'N/A')} months\n"
                    md_content += f"- **Description**: {job.get('description', 'N/A')}\n\n"
                    
                # Skills
                md_content += "## Skills\n"
                for skill in cand.get('skills', []):
                    md_content += f"- **{skill.get('name', 'N/A')}** | {skill.get('proficiency', 'N/A')}\n"
                    
                with open(filepath, 'w', encoding='utf-8') as out_f:
                    out_f.write(md_content)
                    
                print(f"Generated -> {filepath}")
                found_count += 1
                
                if found_count == len(target_ids):
                    break
                    
    print(f"\nSuccessfully generated {found_count} Markdown files in '{out_dir}/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("shift_csv")
    args = parser.parse_args()
    generate_shift_mds(args.shift_csv)
