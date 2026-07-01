import datetime

class TimelineValidator:
    def __init__(self):
        self.today = datetime.datetime.now()

    def _parse_date(self, date_str):
        if not date_str or str(date_str).lower() == 'present':
            return self.today
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return self.today

    def extract_timeline_string(self, candidate):
        """
        Calculates gaps and total durations, returning a structured metadata string.
        """
        edu_metadata = self._get_education_metadata(candidate.get("education", []))
        career_metadata = self._get_career_metadata(candidate.get("career", []))
        
        return f"[TIMELINE_METADATA]\n{edu_metadata}\n{career_metadata}\n[/TIMELINE_METADATA]"

    def _get_education_metadata(self, education_list):
        if not education_list:
            return "[EDUCATION] No formal education listed."
            
        # Sort by start year
        sorted_edu = sorted(education_list, key=lambda x: x.get("start_year", 9999))
        
        total_years = 0
        longest_gap = 0
        
        for i in range(len(sorted_edu)):
            start = sorted_edu[i].get("start_year")
            end = sorted_edu[i].get("end_year")
            if start and end:
                total_years += max(0, end - start)
                
            if i > 0:
                prev_end = sorted_edu[i-1].get("end_year")
                curr_start = start
                if prev_end and curr_start:
                    gap = max(0, curr_start - prev_end)
                    if gap > longest_gap:
                        longest_gap = gap
                        
        return f"[EDUCATION] Total Duration: {total_years} Years | Longest Gap Between Degrees: {longest_gap} Years"

    def _get_career_metadata(self, career_list):
        if not career_list:
            return "[CAREER] No professional experience listed."
            
        # Sort by start date
        sorted_career = sorted(career_list, key=lambda x: self._parse_date(x.get("start_date")))
        
        total_months = 0
        longest_gap_months = 0
        
        for i in range(len(sorted_career)):
            start_d = self._parse_date(sorted_career[i].get("start_date"))
            end_d = self._parse_date(sorted_career[i].get("end_date"))
            
            duration = max(0, (end_d.year - start_d.year) * 12 + (end_d.month - start_d.month))
            total_months += duration
            
            if i > 0:
                prev_end = self._parse_date(sorted_career[i-1].get("end_date"))
                curr_start = start_d
                
                gap = max(0, (curr_start.year - prev_end.year) * 12 + (curr_start.month - prev_end.month))
                if gap > longest_gap_months:
                    longest_gap_months = gap
                    
        total_years = round(total_months / 12.0, 1)
        longest_gap_years = round(longest_gap_months / 12.0, 1)
        
        return f"[CAREER] Total Relevant Experience: {total_years} Years | Longest Unexplained Gap: {longest_gap_years} Years"
