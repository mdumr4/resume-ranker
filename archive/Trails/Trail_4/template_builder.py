import json
import pandas as pd
from datetime import datetime

def calculate_months_between(start_str, end_str):
    if not start_str: return 0
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d") if end_str else datetime.now()
    return (end.year - start.year) * 12 + (end.month - start.month)

def build_templates(jsonl_path, output_parquet):
    print("Building Grammatical Templates from JSONL...")
    records = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            cand = json.loads(line)
            cid = cand['candidate_id']
            
            # --- 1. Career Chunk ---
            career_parts = []
            career_parts.append(f"The candidate has {cand['profile'].get('years_of_experience', 0)} total years of experience.")
            
            # Calculate gap after education
            edu_end_year = 0
            if cand.get('education'):
                edu_end_year = max([e.get('end_year', 0) for e in cand['education'] if e.get('end_year')])
                
            ch = cand.get('career_history', [])
            if ch and edu_end_year > 0:
                first_job_start = ch[-1].get('start_date')
                if first_job_start:
                    start_yr = int(first_job_start.split('-')[0])
                    gap_years = start_yr - edu_end_year
                    if gap_years > 0:
                        career_parts.append(f"After education, there was a career gap of {gap_years * 12} months.")
            
            for idx, role in enumerate(ch):
                desc = role.get('description', '')
                is_curr = "Currently Active" if role.get('is_current') else "Concluded"
                career_parts.append(f"They worked as a {role.get('title')} at {role.get('company')} in the {role.get('industry', 'General')} industry for {role.get('duration_months', 0)} months doing {desc}. This role is {is_curr}.")
                
                # Gap before next role (which is previous in list since list is reverse chronological)
                if idx > 0:
                    prev_start = ch[idx-1].get('start_date')
                    curr_end = role.get('end_date')
                    if prev_start and curr_end:
                        gap = calculate_months_between(curr_end, prev_start)
                        if gap > 1:
                            career_parts.append(f"There was a {gap}-month employment gap before their next role.")
            
            text_career = " ".join(career_parts)
            
            # --- 2. Skills Chunk ---
            skills_parts = []
            for s in cand.get('skills', []):
                skills_parts.append(f"The candidate is highly proficient in {s.get('name')} with {s.get('duration_months', 0)} months of experience.")
            
            for c in cand.get('certifications', []):
                skills_parts.append(f"They hold a {c.get('name')} certification achieved in {c.get('year')}.")
                
            for l in cand.get('languages', []):
                skills_parts.append(f"They speak {l.get('language')} at a {l.get('proficiency')} level.")
                
            sas = cand.get('redrob_signals', {}).get('skill_assessment_scores', {})
            for sk, sc in sas.items():
                skills_parts.append(f"They have a skill assessment score of {sc}/100 in {sk}.")
                
            text_skills = " ".join(skills_parts)
            
            # --- 3. Profile Chunk ---
            prof = cand.get('profile', {})
            sig = cand.get('redrob_signals', {})
            
            prof_parts = []
            prof_parts.append(f"Summary: {prof.get('summary', '')}")
            prof_parts.append(f"Candidate is located in {prof.get('location', '')}, {prof.get('country', '')}.")
            
            sal = sig.get('expected_salary_range_inr_lpa', {})
            prof_parts.append(f"Their expected salary is {sal.get('min', 0)}-{sal.get('max', 0)} LPA.")
            
            reloc = "Willing" if sig.get('willing_to_relocate') else "Unwilling"
            prof_parts.append(f"If the job is outside their city, they are {reloc} to relocate.")
            
            otw = "Open to Work" if sig.get('open_to_work_flag') else "Not actively looking"
            prof_parts.append(f"They are currently {otw} with a notice period of {sig.get('notice_period_days', 0)} days.")
            
            prof_parts.append(f"Their profile is {sig.get('profile_completeness_score', 0)}% complete, with verified contact info and a connected LinkedIn.")
            prof_parts.append(f"Behaviorally, their GitHub score is {sig.get('github_activity_score', 0)} and their recruiter response rate is {sig.get('recruiter_response_rate', 0)*100}%.")
            
            text_profile = " ".join(prof_parts)
            
            # --- 4. Education Chunk ---
            edu_parts = []
            for e in cand.get('education', []):
                edu_parts.append(f"Candidate holds a {e.get('degree')} in {e.get('field_of_study')} from a {e.get('tier')} university, graduating with a CGPA/Grade of {e.get('grade', 'N/A')}.")
            text_edu = " ".join(edu_parts)
            
            # --- Trust Algorithms (RedRob Signals) ---
            # --- 3. Stage 3 Date Math Verification ---
            trust_date_violation = False
            prev_end = None
            
            exp_list = cand.get('career_history', [])
            for exp in exp_list:
                start_str = exp.get('start_date')
                end_str = exp.get('end_date')
                
                if start_str:
                    try:
                        start_date = datetime.strptime(start_str, "%Y-%m-%d")
                        if prev_end and start_date and start_date < prev_end:
                            trust_date_violation = True
                        if end_str:
                            prev_end = datetime.strptime(end_str, "%Y-%m-%d")
                        else:
                            prev_end = datetime.now()
                    except:
                        pass
            
            # --- 5. Timeline Integrity Audit ---
            latest_edu_end_year = None
            for edu in cand.get('education', []):
                end_yr = edu.get('end_year')
                if end_yr and (latest_edu_end_year is None or end_yr > latest_edu_end_year):
                    latest_edu_end_year = end_yr
                    
            earliest_job_start_year = None
            exp_list = cand.get('career_history', [])
            for exp in exp_list:
                start_str = exp.get('start_date')
                if start_str:
                    try:
                        start_yr = datetime.strptime(start_str, "%Y-%m-%d").year
                        if earliest_job_start_year is None or start_yr < earliest_job_start_year:
                            earliest_job_start_year = start_yr
                    except:
                        pass
                        
            trust_career_gap_violation = False
            if latest_edu_end_year and earliest_job_start_year:
                gap = earliest_job_start_year - latest_edu_end_year
                if gap > 2:
                    trust_career_gap_violation = True

            # --- Structural Base Columns ---
            records.append({
                'candidate_id': cid,
                'text_career': text_career,
                'text_skills': text_skills,
                'text_profile': text_profile,
                'text_edu': text_edu,
                'years_of_experience': cand.get('profile', {}).get('total_experience_years', 0),
                'notice_period': cand.get('profile', {}).get('preferences', {}).get('notice_period_days', 0),
                'trust_date_violation': trust_date_violation,
                'trust_career_gap_violation': trust_career_gap_violation
            })
            
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1} candidates...")
                
    df = pd.DataFrame(records)
    df.to_parquet(output_parquet, index=False)
    print(f"Saved {len(df)} templates to {output_parquet}")

if __name__ == "__main__":
    build_templates("candidates.jsonl", "index/features_v2.parquet")
