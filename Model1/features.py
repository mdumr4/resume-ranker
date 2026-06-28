import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CandidateFeatureExtractor:
    """
    Extracts structured tabular features from raw candidate JSON.
    These features are JD-agnostic and focus on behavioral signals, trust metrics,
    and structured profiling to be consumed by LambdaMART.
    """
    def __init__(self):
        # Known description templates can be added here for tier mapping
        pass

    def extract_features(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all pre-computable features from a candidate record."""
        features = {'candidate_id': candidate.get('candidate_id')}
        
        # 1. Structural Features
        features.update(self._extract_structural(candidate))
        
        # 2. Behavioral Signals
        features.update(self._extract_behavioral(candidate))
        
        # 3. Trust & Integrity Features
        features.update(self._extract_trust(candidate))
        
        # 4. Aggregated Skill Metrics
        features.update(self._extract_skill_metrics(candidate))
        
        return features

    def _extract_structural(self, c: Dict[str, Any]) -> Dict[str, Any]:
        profile = c.get('profile', {})
        history = c.get('career_history', [])
        edu = c.get('education', [])
        
        # Calculate actual tenure vs stated experience
        total_tenure_months = sum(job.get('duration_months', 0) for job in history)
        
        # Safely parse company size ordinal
        size_map = {"1-10": 1, "11-50": 2, "51-200": 3, "201-500": 4, 
                    "501-1000": 5, "1001-5000": 6, "5001-10000": 7, "10001+": 8}
        company_size_str = profile.get('current_company_size', '')
        
        return {
            'years_of_experience': float(profile.get('years_of_experience', 0)),
            'total_career_months': total_tenure_months,
            'num_career_entries': len(history),
            'avg_tenure_months': total_tenure_months / max(len(history), 1),
            'max_tenure_months': max([job.get('duration_months', 0) for job in history], default=0),
            'current_company_size_ordinal': size_map.get(company_size_str, 0),
            'num_degrees': len(edu),
            'is_india_based': 1 if profile.get('country', '').lower() == 'india' else 0,
        }

    def _extract_behavioral(self, c: Dict[str, Any]) -> Dict[str, Any]:
        sigs = c.get('redrob_signals', {})
        
        # Handle -1 values for optional metrics
        github = sigs.get('github_activity_score', -1)
        offer_acc = sigs.get('offer_acceptance_rate', -1)
        
        return {
            'profile_completeness': float(sigs.get('profile_completeness_score', 0)),
            'open_to_work': 1 if sigs.get('open_to_work_flag', False) else 0,
            'response_rate': float(sigs.get('recruiter_response_rate', 0)),
            'response_time_hours': float(sigs.get('avg_response_time_hours', 1000)), # high default if missing
            'notice_period_days': int(sigs.get('notice_period_days', 90)),
            'github_score': float(github) if github != -1 else 0.0,
            'has_github': 1 if github != -1 else 0,
            'interview_completion': float(sigs.get('interview_completion_rate', 0)),
            'offer_acceptance': float(offer_acc) if offer_acc != -1 else 0.0,
            'market_demand': int(sigs.get('profile_views_received_30d', 0)) + \
                             int(sigs.get('search_appearance_30d', 0)) + \
                             int(sigs.get('saved_by_recruiters_30d', 0)),
            'connection_count': int(sigs.get('connection_count', 0)),
            'endorsements_received': int(sigs.get('endorsements_received', 0)),
        }

    def _extract_trust(self, c: Dict[str, Any]) -> Dict[str, Any]:
        profile = c.get('profile', {})
        sigs = c.get('redrob_signals', {})
        history = c.get('career_history', [])
        skills = c.get('skills', [])
        
        # 1. Experience mismatch
        stated_exp_months = float(profile.get('years_of_experience', 0)) * 12
        actual_exp_months = sum(job.get('duration_months', 0) for job in history)
        exp_mismatch_months = abs(stated_exp_months - actual_exp_months)
        
        # 2. Date violations
        try:
            signup = datetime.strptime(sigs.get('signup_date', '2030-01-01'), '%Y-%m-%d')
            last_active = datetime.strptime(sigs.get('last_active_date', '1970-01-01'), '%Y-%m-%d')
            date_violation = 1 if last_active < signup else 0
        except ValueError:
            date_violation = 0
            
        # 3. Salary violations
        salary = sigs.get('expected_salary_range_inr_lpa', {})
        min_sal = salary.get('min', 0)
        max_sal = salary.get('max', 0)
        salary_violation = 1 if min_sal > max_sal else 0
        
        # 4. Verification
        both_unverified = 1 if not sigs.get('verified_email') and not sigs.get('verified_phone') else 0
        
        # 5. Skill Stuffing Indicator
        # How many expert skills does the user claim vs their total experience?
        expert_skills = sum(1 for s in skills if s.get('proficiency') == 'expert')
        expert_stuffing_risk = 1 if (expert_skills > 5 and stated_exp_months < 36) else 0

        # Note: Career/Skill text mismatch will be handled via Cross-Encoder in the pipeline,
        # but these structural flags help LambdaMART penalize bad actors.
        
        return {
            'trust_exp_mismatch_months': exp_mismatch_months,
            'trust_date_violation': date_violation,
            'trust_salary_violation': salary_violation,
            'trust_unverified': both_unverified,
            'trust_expert_stuffing_risk': expert_stuffing_risk
        }
        
    def _extract_skill_metrics(self, c: Dict[str, Any]) -> Dict[str, Any]:
        skills = c.get('skills', [])
        
        prof_map = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
        proficiencies = [prof_map.get(s.get('proficiency', ''), 1) for s in skills]
        durations = [s.get('duration_months', 0) for s in skills]
        
        return {
            'num_skills_listed': len(skills),
            'avg_skill_proficiency': float(np.mean(proficiencies)) if proficiencies else 0.0,
            'max_skill_proficiency': float(np.max(proficiencies)) if proficiencies else 0.0,
            'avg_skill_duration_months': float(np.mean(durations)) if durations else 0.0,
        }

if __name__ == "__main__":
    import os
    # Quick test
    sample_path = "../candidates.jsonl" if not os.path.exists("Sample/sample_candidates.json") else "Sample/sample_candidates.json"
    
    try:
        if sample_path.endswith(".jsonl"):
            with open(sample_path, 'r', encoding='utf-8') as f:
                c = json.loads(f.readline())
        else:
            with open(sample_path, 'r', encoding='utf-8') as f:
                c = json.load(f)[0]
                
        extractor = CandidateFeatureExtractor()
        features = extractor.extract_features(c)
        print("Feature extraction successful.")
        for k, v in features.items():
            print(f"{k}: {v}")
    except Exception as e:
        print(f"Could not run test: {e}")
