import json
import pandas as pd
from datetime import datetime
import numpy as np
import os
from tqdm import tqdm

def parse_date(d_str):
    if not d_str: return None
    try:
        return datetime.strptime(d_str, "%Y-%m-%d")
    except:
        return None

def build_templates(jsonl_path="candidates.jsonl", output_parquet="index/features_v2.parquet"):
    """
    Trail 6 Stage 0 Ingestion: 
    Converts raw JSON into highly enriched Semantic Text Templates and calculates the Stage 1 Trust Score.
    """
    records = []
    
    # We use a fixed Max Date to simulate "today" for platform tenure calculations
    MAX_SYSTEM_DATE = datetime(2026, 5, 27)
    
    print(f"Reading candidates from {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            cand = json.loads(line)
            cid = cand.get('candidate_id')
            prof = cand.get('profile', {})
            ch = cand.get('career_history', [])
            edu = cand.get('education', [])
            skills = cand.get('skills', [])
            certs = cand.get('certifications', [])
            langs = cand.get('languages', [])
            sig = cand.get('redrob_signals', {})
            
            # ---------------------------------------------------------
            # 1. TEXT_PROFILE ENHANCEMENT
            # ---------------------------------------------------------
            yoe = prof.get('years_of_experience', 0)
            country = prof.get('country', 'Unknown')
            
            # JD-Dependent Signals extracted to semantic text
            prefs = prof.get('preferences', {})
            notice = prefs.get('notice_period_days', 0)
            work_mode = prefs.get('preferred_work_mode', 'flexible')
            relocate = prefs.get('willing_to_relocate', False)
            sal = prefs.get('expected_salary_range_inr_lpa', {})
            sal_min = sal.get('min', 0)
            sal_max = sal.get('max', 0)
            
            p_parts = []
            if prof.get('headline'): p_parts.append(prof['headline'])
            if prof.get('summary'): p_parts.append(prof['summary'])
            if prof.get('location'): p_parts.append(f"Located in {prof['location']}, {country}.")
            
            p_parts.append(f"Candidate expects {sal_min}-{sal_max} LPA, prefers {work_mode} work mode, notice period is {notice} days, willing to relocate: {relocate}.")
            text_profile = " ".join(p_parts)
            
            # ---------------------------------------------------------
            # 2. TEXT_CAREER ENHANCEMENT
            # ---------------------------------------------------------
            c_parts = [f"Candidate has a total of {yoe} years of experience."]
            
            for exp in ch:
                title = exp.get('title', '')
                comp = exp.get('company', '')
                dur = exp.get('duration_months', 0)
                ind = exp.get('industry', 'various')
                desc = exp.get('description', '')
                
                if exp.get('is_current', False):
                    c_parts.append(f"Currently working as a {title} at {comp} in the {ind} industry.")
                else:
                    c_parts.append(f"Worked as a {title} at {comp} for {dur} months in the {ind} industry.")
                    
                if desc: c_parts.append(desc)
                
            text_career = " ".join(c_parts)
            
            # ---------------------------------------------------------
            # 3. TEXT_EDU ENHANCEMENT
            # ---------------------------------------------------------
            e_parts = []
            for ed in edu:
                deg = ed.get('degree', '')
                fld = ed.get('field_of_study', '')
                inst = ed.get('institution', '')
                sy = ed.get('start_year', '')
                ey = ed.get('end_year', '')
                grade = ed.get('grade', 'N/A')
                tier = ed.get('tier', 'unknown')
                
                e_parts.append(f"Holds a {deg} in {fld} from {inst} ({sy} - {ey}).")
                e_parts.append(f"Achieved a grade of {grade} from a {tier} institution.")
                
            text_edu = " ".join(e_parts)
            
            # ---------------------------------------------------------
            # 4. TEXT_SKILLS ENHANCEMENT (Verification Formula)
            # ---------------------------------------------------------
            s_parts = []
            for s in skills:
                name = s.get('name', '')
                prof_lvl = s.get('proficiency', '')
                end = s.get('endorsements', 0)
                dur = s.get('duration_months', 0)
                
                if dur > 24 and end > 10:
                    s_parts.append(f"Highly verified {prof_lvl} in {name}.")
                elif end > 0:
                    s_parts.append(f"Verified {prof_lvl} in {name}.")
                else:
                    s_parts.append(f"Claims {prof_lvl} in {name}.")
                    
            for c in certs:
                s_parts.append(f"Certified in {c.get('name')} by {c.get('issuer')} ({c.get('year')}).")
                
            for l in langs:
                s_parts.append(f"Speaks {l.get('language')} ({l.get('proficiency')}).")
                
            text_skills = " ".join(s_parts)
            
            # ---------------------------------------------------------
            # 5. DYNAMIC TRUST & ACTIVENESS MATH ENGINE
            # ---------------------------------------------------------
            v_email = sig.get('verified_email', False)
            v_phone = sig.get('verified_phone', False)
            
            # Career Gap Calculation
            latest_edu = None
            for ed in edu:
                yr = ed.get('end_year')
                if yr and (latest_edu is None or yr > latest_edu):
                    latest_edu = yr
                    
            earliest_job = None
            for exp in ch:
                d = parse_date(exp.get('start_date'))
                if d and (earliest_job is None or d.year < earliest_job):
                    earliest_job = d.year
                    
            gap_penalty = 0.0
            if latest_edu and earliest_job:
                gap = earliest_job - latest_edu
                if gap > 0:
                    gap_penalty = min(gap, 15) / 15 * 0.30  # Max -0.30
                    
            # Date Anomaly / Inactivity Calculation
            signup_date = parse_date(sig.get('signup_date'))
            active_date = parse_date(sig.get('last_active_date'))
            date_penalty = 0.0
            
            if signup_date and active_date:
                platform_tenure = (active_date - signup_date).days
                days_since_active = (MAX_SYSTEM_DATE - active_date).days
                
                if platform_tenure < 0 or days_since_active > 200:
                    date_penalty = 0.05
                else:
                    date_penalty = min(days_since_active, 200) / 200 * 0.05
            
            # Core Math
            if not v_email and not v_phone:
                trust_score = 0.10
            else:
                score = 0.0
                
                # High (Max 0.75)
                score += (sig.get('profile_completeness_score', 0) / 100) * 0.20
                if sig.get('open_to_work_flag', False): score += 0.20
                score += max(0, (280 - sig.get('avg_response_time_hours', 280)) / 280) * 0.15
                score += (sig.get('interview_completion_rate', 0) / 1.0) * 0.10
                oar = sig.get('offer_acceptance_rate', -1)
                if oar >= 0: score += (oar / 1.0) * 0.10
                
                # Medium (Max 0.20)
                score += (sig.get('recruiter_response_rate', 0) / 1.0) * 0.10
                score += min(sig.get('search_appearance_30d', 0), 226) / 226 * 0.10
                
                # Bonus (Max 0.05)
                gh = sig.get('github_activity_score', -1)
                if gh >= 0: score += min(gh, 50) / 50 * 0.05
                
                # Penalties
                score -= date_penalty
                score -= gap_penalty
                
                # Clamp
                trust_score = max(0.1, min(1.0, score))
            
            # Output Row
            records.append({
                'candidate_id': cid,
                'text_career': text_career,
                'text_profile': text_profile,
                'text_skills': text_skills,
                'text_edu': text_edu,
                'trust_score': float(trust_score)
            })
            
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1} candidates...")
                
    print("Writing to parquet...")
    df = pd.DataFrame(records)
    
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"Successfully wrote {len(df)} records to {output_parquet}")
    print(df[['candidate_id', 'trust_score']].head(10))

if __name__ == "__main__":
    # If running locally in repo root
    build_templates(jsonl_path="../../candidates.jsonl", output_parquet="../../index/features_v3.parquet")
