import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd

logger = logging.getLogger("preprocessor")


def parse_date(d_str: Any) -> Optional[datetime]:
    """Safely parses YYYY-MM-DD date strings."""
    if not d_str or pd.isna(d_str):
        return None
    try:
        return datetime.strptime(str(d_str).strip(), "%Y-%m-%d")
    except Exception:
        return None


class CandidatePreprocessor:
    """
    Ingests and serializes candidate profiles into enriched semantic text
    and computes the Multi-Factor Dynamic Trust Score.
    """

    def __init__(self, max_system_date: datetime = datetime(2026, 5, 27)):
        self.max_system_date = max_system_date

    def serialize_candidate(self, cand: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a raw JSON candidate dictionary into four semantic text chunks
        and calculates the trust score.
        """
        cid = cand.get("candidate_id", "UNKNOWN")
        prof = cand.get("profile", {})
        ch = cand.get("career_history", [])
        edu = cand.get("education", [])
        skills = cand.get("skills", [])
        certs = cand.get("certifications", [])
        langs = cand.get("languages", [])
        sig = cand.get("redrob_signals", {})

        # 1. TEXT_PROFILE ENHANCEMENT
        yoe = prof.get("years_of_experience", 0)
        country = prof.get("country", "Unknown")
        prefs = prof.get("preferences", {})
        notice = prefs.get("notice_period_days", 0)
        work_mode = prefs.get("preferred_work_mode", "flexible")
        relocate = prefs.get("willing_to_relocate", False)
        sal = prefs.get("expected_salary_range_inr_lpa", {})
        sal_min = sal.get("min", 0)
        sal_max = sal.get("max", 0)

        p_parts = []
        if prof.get("headline"):
            p_parts.append(str(prof["headline"]))
        if prof.get("summary"):
            p_parts.append(str(prof["summary"]))
        if prof.get("location"):
            p_parts.append(f"Located in {prof['location']}, {country}.")
        p_parts.append(
            f"Candidate expects {sal_min}-{sal_max} LPA, prefers {work_mode} work mode, "
            f"notice period is {notice} days, willing to relocate: {relocate}."
        )
        text_profile = " ".join(p_parts)

        # 2. TEXT_CAREER ENHANCEMENT
        c_parts = [f"Candidate has a total of {yoe} years of experience."]
        for exp in ch:
            title = exp.get("title", "")
            comp = exp.get("company", "")
            dur = exp.get("duration_months", 0)
            ind = exp.get("industry", "various")
            desc = exp.get("description", "")

            if exp.get("is_current", False):
                c_parts.append(f"Currently working as a {title} at {comp} in the {ind} industry.")
            else:
                c_parts.append(f"Worked as a {title} at {comp} for {dur} months in the {ind} industry.")

            if desc:
                c_parts.append(str(desc))
        text_career = " ".join(c_parts)

        # 3. TEXT_EDU ENHANCEMENT
        e_parts = []
        for ed in edu:
            deg = ed.get("degree", "")
            fld = ed.get("field_of_study", "")
            inst = ed.get("institution", "")
            sy = ed.get("start_year", "")
            ey = ed.get("end_year", "")
            grade = ed.get("grade", "N/A")
            tier = ed.get("tier", "unknown")

            e_parts.append(f"Holds a {deg} in {fld} from {inst} ({sy} - {ey}).")
            e_parts.append(f"Achieved a grade of {grade} from a {tier} institution.")
        text_edu = " ".join(e_parts)

        # 4. TEXT_SKILLS ENHANCEMENT
        s_parts = []
        for s in skills:
            name = s.get("name", "")
            prof_lvl = s.get("proficiency", "")
            end = s.get("endorsements", 0)
            dur = s.get("duration_months", 0)

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

        # 5. DYNAMIC TRUST & ACTIVENESS MATH ENGINE
        trust_score = self.calculate_trust_score(sig, ch, edu)

        return {
            "candidate_id": cid,
            "text_career": text_career,
            "text_profile": text_profile,
            "text_skills": text_skills,
            "text_edu": text_edu,
            "trust_score": float(trust_score),
            "experience_years": yoe,
        }

    def calculate_trust_score(
        self, sig: Dict[str, Any], ch: List[Dict[str, Any]], edu: List[Dict[str, Any]]
    ) -> float:
        """Calculates multi-factor trust score with merged timeline gap detection."""
        v_email = sig.get("verified_email", False)
        v_phone = sig.get("verified_phone", False)

        # Career Gap Calculation - Merged Timeline Interval Approach
        intervals: List[Tuple[int, int]] = []

        # Extract education intervals
        for ed in edu:
            sy = ed.get("start_year")
            ey = ed.get("end_year")
            if sy is not None and ey is not None:
                try:
                    intervals.append((int(sy), int(ey)))
                except Exception:
                    pass

        # Extract career intervals
        for exp in ch:
            sd = parse_date(exp.get("start_date"))
            ed = parse_date(exp.get("end_date"))
            is_curr = exp.get("is_current", False)
            if sd:
                s_year = sd.year
                e_year = self.max_system_date.year if (is_curr or not ed) else ed.year
                intervals.append((s_year, e_year))

        gap_penalty = 0.0
        if intervals:
            intervals.sort(key=lambda x: x[0])
            merged: List[Tuple[int, int]] = []
            for interval in intervals:
                if not merged:
                    merged.append(interval)
                else:
                    last_s, last_e = merged[-1]
                    curr_s, curr_e = interval
                    if curr_s <= last_e:
                        merged[-1] = (last_s, max(last_e, curr_e))
                    else:
                        merged.append(interval)

            total_gap = 0
            for idx in range(len(merged) - 1):
                gap = merged[idx + 1][0] - merged[idx][1]
                if gap > 1:
                    total_gap += gap - 1

            gap_penalty = min(total_gap, 15) / 15 * 0.45

        # Date Inactivity Calculation
        signup_date = parse_date(sig.get("signup_date"))
        active_date = parse_date(sig.get("last_active_date"))
        date_penalty = 0.0

        if signup_date and active_date:
            platform_tenure = (active_date - signup_date).days
            days_since_active = (self.max_system_date - active_date).days

            if platform_tenure < 0 or days_since_active > 200:
                date_penalty = 0.05
            else:
                date_penalty = min(days_since_active, 200) / 200 * 0.05

        # Verification Gate
        if not v_email and not v_phone:
            return 0.10

        score = 0.0
        # High-weight signals (Max 0.75)
        score += (sig.get("profile_completeness_score", 0) / 100) * 0.20
        if sig.get("open_to_work_flag", False):
            score += 0.20
        score += max(0, (280 - sig.get("avg_response_time_hours", 280)) / 280) * 0.15
        score += (sig.get("interview_completion_rate", 0) / 1.0) * 0.10
        oar = sig.get("offer_acceptance_rate", -1)
        if oar >= 0:
            score += (oar / 1.0) * 0.10

        # Medium-weight signals (Max 0.20)
        score += (sig.get("recruiter_response_rate", 0) / 1.0) * 0.10
        score += min(sig.get("search_appearance_30d", 0), 226) / 226 * 0.10

        # Bonus (Max 0.05)
        gh = sig.get("github_activity_score", -1)
        if gh >= 0:
            score += min(gh, 50) / 50 * 0.05

        # Penalties
        score -= date_penalty
        score -= gap_penalty

        return float(max(0.1, min(1.0, score)))

    def process_file(
        self, input_path: Union[str, Path], output_parquet: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Processes JSON or JSONL file into a features DataFrame."""
        path = Path(input_path)
        records = []

        logger.info(f"Reading candidates from {path}...")
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for cand in data:
                    records.append(self.serialize_candidate(cand))
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        cand = json.loads(line)
                        records.append(self.serialize_candidate(cand))

        df = pd.DataFrame(records)
        logger.info(f"Processed {len(df)} candidate records.")

        if output_parquet:
            out_p = Path(output_parquet)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_p, index=False)
            logger.info(f"Saved features to {out_p}")

        return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Candidate Ingestion & Dynamic Trust Score Preprocessor")
    parser.add_argument("--input", default="data/candidates.jsonl", help="Path to input candidates.jsonl")
    parser.add_argument("--output", default="index/features_v3.parquet", help="Path to output features.parquet")
    args = parser.parse_args()

    prep = CandidatePreprocessor()
    prep.process_file(args.input, args.output)
