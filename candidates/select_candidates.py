import os
import json
import random

def reservoir_sample(filename, k=10):
    """Selects k lines randomly from a file using reservoir sampling."""
    sample = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i < k:
                sample.append(line)
            else:
                r = random.randint(0, i)
                if r < k:
                    sample[r] = line
    return sample

def format_candidate_md(data):
    profile = data.get("profile", {})
    name = profile.get("anonymized_name", "Unknown Candidate")
    headline = profile.get("headline", "No headline provided")
    summary = profile.get("summary", "No summary provided")
    location = profile.get("location", "Unknown Location")
    country = profile.get("country", "Unknown Country")
    years_exp = profile.get("years_of_experience", "N/A")
    candidate_id = data.get("candidate_id", "N/A")

    md = []
    md.append(f"# Candidate: {name}")
    md.append(f"**Candidate ID**: {candidate_id}")
    md.append(f"**Headline**: {headline}")
    md.append(f"**Location**: {location}, {country} | **Experience**: {years_exp} Years\n")
    
    md.append("## Summary")
    md.append(f"{summary}\n")
    
    md.append("## Skills")
    skills = data.get("skills", [])
    if skills:
        for skill in skills:
            s_name = skill.get("name", "N/A")
            prof = skill.get("proficiency", "N/A")
            endors = skill.get("endorsements", 0)
            dur = skill.get("duration_months", 0)
            md.append(f"- **{s_name}** ({prof}) — {endors} endorsement(s), {dur} month(s)")
    else:
        md.append("No skills listed.")
    md.append("")

    md.append("## Career History")
    history = data.get("career_history", [])
    if history:
        for job in history:
            company = job.get("company", "N/A")
            title = job.get("title", "N/A")
            start = job.get("start_date", "N/A")
            end = job.get("end_date") or "Present"
            dur = job.get("duration_months", "N/A")
            desc = job.get("description", "No description provided.")
            md.append(f"### {title} at {company}")
            md.append(f"**Duration**: {start} to {end} ({dur} months) | **Current**: {job.get('is_current', False)}")
            md.append(f"{desc}\n")
    else:
        md.append("No career history listed.\n")

    md.append("## Education")
    education = data.get("education", [])
    if education:
        for edu in education:
            inst = edu.get("institution", "N/A")
            deg = edu.get("degree", "N/A")
            field = edu.get("field_of_study", "N/A")
            start = edu.get("start_year", "N/A")
            end = edu.get("end_year", "N/A")
            grade = edu.get("grade", "N/A")
            tier = edu.get("tier", "N/A")
            md.append(f"- **{deg}** in *{field}*, {inst} ({start} - {end})")
            md.append(f"  - Grade: {grade} | Tier: {tier}")
    else:
        md.append("No education listed.")
    md.append("")

    md.append("## Certifications")
    certs = data.get("certifications", [])
    if certs:
        for cert in certs:
            c_name = cert.get("name", "N/A")
            issuer = cert.get("issuer", "N/A")
            year = cert.get("year", "N/A")
            md.append(f"- **{c_name}** issued by {issuer} ({year})")
    else:
        md.append("No certifications listed.")
    md.append("")

    md.append("## Languages")
    langs = data.get("languages", [])
    if langs:
        for lang in langs:
            l_name = lang.get("language", "N/A")
            prof = lang.get("proficiency", "N/A")
            md.append(f"- **{l_name}** ({prof})")
    else:
        md.append("No languages listed.")
    md.append("")

    md.append("## Recruiter & Platform Signals")
    signals = data.get("redrob_signals", {})
    if signals:
        expected_salary = signals.get("expected_salary_range_inr_lpa", {})
        min_sal = expected_salary.get("min", "N/A")
        max_sal = expected_salary.get("max", "N/A")
        md.append(f"- **Profile Completeness**: {signals.get('profile_completeness_score', 'N/A')}%")
        md.append(f"- **Open to Work**: {signals.get('open_to_work_flag', 'N/A')}")
        md.append(f"- **Notice Period**: {signals.get('notice_period_days', 'N/A')} days")
        md.append(f"- **Expected Salary Range**: {min_sal} - {max_sal} INR LPA")
        md.append(f"- **Preferred Work Mode**: {signals.get('preferred_work_mode', 'N/A')}")
        md.append(f"- **Willing to Relocate**: {signals.get('willing_to_relocate', 'N/A')}")
        md.append(f"- **GitHub Activity Score**: {signals.get('github_activity_score', 'N/A')}")
        md.append(f"- **Interview Completion Rate**: {signals.get('interview_completion_rate', 'N/A')}")
        md.append(f"- **Offer Acceptance Rate**: {signals.get('offer_acceptance_rate', 'N/A')}")
    else:
        md.append("No signals available.")
    md.append("")

    return name, "\n".join(md)

def main():
    jsonl_file = "candidates.jsonl"
    output_dir = "candidates"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    print("Selecting 10 random candidates using reservoir sampling...")
    sampled_lines = reservoir_sample(jsonl_file, 10)
    
    for idx, line in enumerate(sampled_lines):
        try:
            data = json.loads(line)
            name, md_content = format_candidate_md(data)
            
            # Clean name for safe filename
            safe_name = "".join([c for c in name if c.isalpha() or c.isspace() or c == '-']).strip()
            if not safe_name:
                safe_name = f"Candidate_{data.get('candidate_id', idx)}"
                
            filename = os.path.join(output_dir, f"{safe_name}.md")
            # If name duplicate exists, add ID
            if os.path.exists(filename):
                filename = os.path.join(output_dir, f"{safe_name}_{data.get('candidate_id')}.md")
                
            with open(filename, 'w', encoding='utf-8') as out_f:
                out_f.write(md_content)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error parsing line {idx}: {e}")

if __name__ == "__main__":
    main()
