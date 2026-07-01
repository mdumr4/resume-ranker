import pandas as pd
import json
import os
import argparse

def generate_mds(trail_csv, jsonl_path="candidates.jsonl", out_dir="candidates"):
    print(f"Loading {trail_csv}...")
    try:
        df = pd.read_csv(trail_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {trail_csv}")
        return

    # Ensure sorted by rank
    df = df.sort_values('rank').reset_index(drop=True)
    
    # Get Top 10 and Bottom 10 (or as many as available)
    top_10 = df.head(10)
    bottom_10 = df.tail(10)
    
    target_ids = {}
    for i, row in top_10.iterrows():
        target_ids[row['candidate_id']] = f"Top_{int(row['rank'])}"
        
    for i, row in bottom_10.iterrows():
        target_ids[row['candidate_id']] = f"Bottom_{int(row['rank'])}"
        
    print(f"Searching for {len(target_ids)} candidates in {jsonl_path}...")
    
    os.makedirs(out_dir, exist_ok=True)
    found_candidates = {}
    
    # Read the massive JSONL file
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            cand = json.loads(line)
            cid = cand.get('candidate_id')
            if cid in target_ids:
                found_candidates[cid] = cand
                if len(found_candidates) == len(target_ids):
                    break
                    
    print(f"Found {len(found_candidates)} candidates. Generating Markdown files...")
    
    for cid, cand in found_candidates.items():
        rank_label = target_ids[cid]
        filename = f"{rank_label}_{cid}.md"
        filepath = os.path.join(out_dir, filename)
        
        # Build beautiful Markdown
        md_content = f"# Candidate: {cid} ({rank_label})\n\n"
        
        # Redrob Signals (The most important for Trust Score)
        sig = cand.get('redrob_signals', {})
        md_content += "## Redrob Signals (Trust Data)\n"
        md_content += f"- **Profile Completeness**: {sig.get('profile_completeness_score', 'N/A')}%\n"
        md_content += f"- **Avg Response Time**: {sig.get('avg_response_time_hours', 'N/A')} hours\n"
        md_content += f"- **Open to Work**: {sig.get('open_to_work_flag', False)}\n"
        md_content += f"- **Interview Completion**: {sig.get('interview_completion_rate', 'N/A')}\n"
        md_content += f"- **Offer Acceptance**: {sig.get('offer_acceptance_rate', 'N/A')}\n"
        md_content += f"- **Verified Email/Phone**: {sig.get('verified_email', False)} / {sig.get('verified_phone', False)}\n"
        md_content += f"- **Signup Date**: {sig.get('signup_date', 'N/A')}\n"
        md_content += f"- **Last Active**: {sig.get('last_active_date', 'N/A')}\n\n"
        
        # Profile
        prof = cand.get('profile', {})
        md_content += "## Profile & Logistics\n"
        md_content += f"- **Location**: {prof.get('location', 'N/A')}, {prof.get('country', 'N/A')}\n"
        md_content += f"- **Total YOE**: {prof.get('years_of_experience', 'N/A')}\n"
        
        sal = sig.get('expected_salary_range_inr_lpa', {})
        md_content += f"- **Expected Salary**: {sal.get('min', 0)} - {sal.get('max', 0)} LPA\n"
        md_content += f"- **Notice Period**: {sig.get('notice_period_days', 'N/A')} days\n"
        md_content += f"- **Work Mode**: {sig.get('preferred_work_mode', 'N/A')}\n"
        md_content += f"- **Willing to Relocate**: {sig.get('willing_to_relocate', False)}\n\n"
        
        # Career
        md_content += "## Career History\n"
        for job in cand.get('career_history', []):
            sd = job.get('start_date', 'Unknown')
            ed = job.get('end_date', 'Present') if job.get('end_date') else 'Present'
            
            md_content += f"### {job.get('title', 'N/A')} at {job.get('company', 'N/A')}\n"
            md_content += f"- **Industry**: {job.get('industry', 'N/A')}\n"
            md_content += f"- **Duration**: {sd} to {ed} ({job.get('duration_months', 'N/A')} months)\n"
            md_content += f"- **Current Role**: {job.get('is_current', False)}\n"
            md_content += f"- **Description**: {job.get('description', 'N/A')}\n\n"
            
        # Education
        md_content += "## Education\n"
        for edu in cand.get('education', []):
            md_content += f"### {edu.get('degree', 'N/A')} in {edu.get('field_of_study', 'N/A')}\n"
            md_content += f"- **Institution**: {edu.get('institution', 'N/A')} (Tier: {edu.get('tier', 'N/A')})\n"
            md_content += f"- **Grade**: {edu.get('grade', 'N/A')}\n"
            md_content += f"- **Years**: {edu.get('start_year', 'N/A')} - {edu.get('end_year', 'N/A')}\n\n"
            
        # Skills
        md_content += "## Skills\n"
        for skill in cand.get('skills', []):
            md_content += f"- **{skill.get('name', 'N/A')}** | {skill.get('proficiency', 'N/A')} | Endorsements: {skill.get('endorsements', 0)} | Months: {skill.get('duration_months', 0)}\n"
            
        with open(filepath, 'w', encoding='utf-8') as out_f:
            out_f.write(md_content)
            
        print(f"Generated -> {filepath}")
        
    print("\nAll requested Markdown files have been successfully generated in the 'candidates/' folder!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Markdown profiles for Top 10 and Bottom 10 candidates of a Trail.")
    parser.add_argument("trail_csv", help="Path to the CSV file containing candidate_id and rank")
    args = parser.parse_args()
    
    generate_mds(args.trail_csv)
