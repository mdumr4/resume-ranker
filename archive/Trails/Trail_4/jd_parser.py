import os
import json

class JDParser:
    def __init__(self, json_path="context/parsed_jd.json"):
        self.json_path = json_path
        
    def parse(self):
        """Returns a dictionary containing the structured JD sections."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Missing pre-processed JD file at {self.json_path}. Please run the LLM prompt and save the JSON first.")
            
        with open(self.json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # 1. Build Semantic Strings for Cross-Encoder
        career_text = f"{raw_data.get('role_title', '')}. {raw_data.get('ideal_candidate_narrative', '')}"
        
        req_skills = [s['name'] for s in raw_data.get('required_skills', [])]
        pref_skills = [s['name'] for s in raw_data.get('preferred_skills', [])]
        skills_text = "Required: " + ", ".join(req_skills) + ". Preferred: " + ", ".join(pref_skills)
        
        profile_text = " ".join(raw_data.get('culture_signals', []))
        edu_text = raw_data.get('education_requirements', {}).get('evidence', '')
        summary_text = raw_data.get('ideal_candidate_narrative', '')
        
        # 2. Extract Constraints for Logistic Filtering
        constraints = {
            "exp_min": raw_data.get('experience_range', {}).get('min_years', 0),
            "exp_max": raw_data.get('experience_range', {}).get('max_years', 99),
            "locations": raw_data.get('logistics', {}).get('preferred_locations', []) + raw_data.get('logistics', {}).get('welcome_locations', []),
            "notice_max": raw_data.get('logistics', {}).get('notice_period', {}).get('preferred_max_days', 30),
            "weights": raw_data.get('section_importance', {"career": 0.35, "skills": 0.25, "profile": 0.20, "education": 0.05})
        }
        
        return {
            "career": career_text,
            "skills": skills_text,
            "profile": profile_text,
            "education": edu_text,
            "summary": summary_text,
            "constraints": constraints
        }

if __name__ == "__main__":
    parser = JDParser("context/parsed_jd.json")
    try:
        jd_data = parser.parse()
        print("Successfully loaded parsed JD!")
        for k, v in jd_data.items():
            print(f"[{k.upper()}] Length: {len(v)} chars")
    except Exception as e:
        print(f"Error: {e}")
