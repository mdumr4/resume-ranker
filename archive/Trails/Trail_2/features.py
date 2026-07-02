import json
import logging
from typing import Dict, Any, Tuple
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class CandidateProcessor:
    """
    Handles both Tabular Feature extraction (for the final math formula) 
    and Text String formatting (for the Cross-Encoder).
    """
    
    def process(self, candidate: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Returns:
            tabular_features: Dict of numerical/boolean features (Trust, Behavior)
            text_blocks: Dict of the 4 formatted strings for the Cross-Encoder
        """
        tabular = self._extract_tabular_features(candidate)
        text_blocks = self._format_text_blocks(candidate)
        return tabular, text_blocks

    def _extract_tabular_features(self, c: Dict[str, Any]) -> Dict[str, float]:
        features = {}
        features['candidate_id'] = c.get('candidate_id')
        
        profile = c.get('profile', {})
        signals = c.get('redrob_signals', {})
        history = c.get('career_history', [])
        skills = c.get('skills', [])
        
        # 1. Structural
        features['years_of_experience'] = float(profile.get('years_of_experience', 0))
        features['is_india_based'] = 1.0 if str(profile.get('country', '')).lower() == 'india' else 0.0
        
        total_months = sum([job.get('duration_months', 0) for job in history])
        features['total_career_months'] = float(total_months)
        
        # 2. Behavioral
        features['profile_completeness'] = float(signals.get('profile_completeness_score', 0))
        features['response_rate'] = float(signals.get('recruiter_response_rate', 0))
        features['notice_period'] = float(signals.get('notice_period_days', 90))
        features['github_score'] = float(signals.get('github_activity_score', 0))
        features['interview_completion'] = float(signals.get('interview_completion_rate', 0))
        
        # 3. Trust / Integrity Flags (Honeypot Catchers)
        # 3a. Experience Mismatch (Claimed vs Actual sum of jobs)
        claimed_months = features['years_of_experience'] * 12
        features['trust_exp_mismatch_months'] = abs(claimed_months - total_months)
        
        # 3b. Salary Logic Violation
        salary = signals.get('expected_salary_range_inr_lpa', {})
        s_min = salary.get('min', 0)
        s_max = salary.get('max', 0)
        features['trust_salary_violation'] = 1.0 if s_min > s_max else 0.0
        
        # 3c. Date Logic Violation (Active before signup)
        features['trust_date_violation'] = 0.0
        try:
            signup = datetime.strptime(signals.get('signup_date', '1970-01-01'), '%Y-%m-%d')
            last_active = datetime.strptime(signals.get('last_active_date', '1970-01-01'), '%Y-%m-%d')
            if last_active < signup:
                features['trust_date_violation'] = 1.0
        except:
            pass # Ignore if dates are missing/malformed
            
        return features

    def _format_text_blocks(self, c: Dict[str, Any]) -> Dict[str, str]:
        profile = c.get('profile', {})
        signals = c.get('redrob_signals', {})
        history = c.get('career_history', [])
        
        # ---------------------------------------------------------
        # Section 1: Career (Focus: Title, Company, Description, Signals)
        # ---------------------------------------------------------
        career_parts = []
        curr_title = profile.get('current_title', '')
        curr_company = profile.get('current_company', '')
        curr_industry = profile.get('current_industry', '')
        exp_years = profile.get('years_of_experience', 0)
        
        career_parts.append(f"Current: {curr_title} at {curr_company} (Industry: {curr_industry}). Total Experience: {exp_years} years.")
        
        # Add top 2 most recent job descriptions
        for i, job in enumerate(history[:2]):
            title = job.get('title', '')
            company = job.get('company', '')
            industry = job.get('industry', '')
            dur = job.get('duration_months', 0)
            desc = job.get('description', '')
            career_parts.append(f"Role {i+1}: {title} at {company} ({dur} months, Industry: {industry}). Responsibilities: {desc}")
            
        oar = signals.get('offer_acceptance_rate', 0) * 100
        icr = signals.get('interview_completion_rate', 0) * 100
        career_parts.append(f"Offer Acceptance Rate: {oar:.1f}%. Interview Completion Rate: {icr:.1f}%.")
        
        # ---------------------------------------------------------
        # Section 2: Skills (Focus: Skills, Duration, Certs, Assessments)
        # ---------------------------------------------------------
        skill_parts = []
        skills = sorted(c.get('skills', []), key=lambda x: {'expert':4, 'advanced':3, 'intermediate':2, 'beginner':1}.get(str(x.get('proficiency')).lower(), 0), reverse=True)
        
        s_list = [f"{s.get('name')} ({s.get('proficiency')}, {s.get('duration_months', 0)}mo)" for s in skills[:15]]
        skill_parts.append(f"Top Skills: {', '.join(s_list)}")
        
        certs = [f"{cert.get('name')} from {cert.get('issuer')} ({cert.get('year')})" for cert in c.get('certifications', [])]
        if certs:
            skill_parts.append(f"Certifications: {'; '.join(certs)}")
            
        assess = signals.get('skill_assessment_scores', {})
        if assess:
            a_list = [f"{k}: {v}%" for k,v in assess.items()]
            skill_parts.append(f"Redrob Assessments Passed: {', '.join(a_list)}")

        # ---------------------------------------------------------
        # Section 3: Profile (Focus: Headline, Summary, Location, Logistics)
        # ---------------------------------------------------------
        profile_parts = []
        profile_parts.append(f"Headline: {profile.get('headline', '')}")
        profile_parts.append(f"Summary: {profile.get('summary', '')}")
        profile_parts.append(f"Location: {profile.get('location', '')}, {profile.get('country', '')}")
        
        sal = signals.get('expected_salary_range_inr_lpa', {})
        profile_parts.append(
            f"Open to work: {signals.get('open_to_work_flag')}. "
            f"Notice Period: {signals.get('notice_period_days')} days. "
            f"Expected Salary: {sal.get('min')}-{sal.get('max')} LPA. "
            f"Preferred Mode: {signals.get('preferred_work_mode')}. "
            f"Willing to relocate: {signals.get('willing_to_relocate')}."
        )

        # ---------------------------------------------------------
        # Section 4: Education
        # ---------------------------------------------------------
        edu_parts = []
        for e in c.get('education', []):
            edu_parts.append(f"{e.get('degree')} in {e.get('field_of_study')} at {e.get('institution')} ({e.get('start_year')}-{e.get('end_year')}). Tier: {e.get('tier')}.")

        return {
            "career": " ".join(career_parts),
            "skills": " ".join(skill_parts),
            "profile": " ".join(profile_parts),
            "education": " ".join(edu_parts)
        }

if __name__ == "__main__":
    # Quick local test
    import os
    if os.path.exists("../candidates.jsonl"):
        with open("../candidates.jsonl", 'r') as f:
            sample_json = json.loads(f.readline())
            processor = CandidateProcessor()
            tab, text = processor.process(sample_json)
            print("--- TABULAR ---")
            for k,v in tab.items(): print(f"{k}: {v}")
            print("\n--- TEXT BLOCKS ---")
            for k,v in text.items(): print(f"{k}:\n{v}\n")
