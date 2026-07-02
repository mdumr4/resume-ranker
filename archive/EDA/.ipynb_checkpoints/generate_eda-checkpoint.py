import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Stage 1: Candidate Data Exploratory Data Analysis (EDA)\n",
                "\n",
                "**Goal**: Thoroughly analyze the `candidates.jsonl` dataset to understand all features and identify heuristics for detecting 'fake' or low-quality candidates. This filtering layer must run entirely offline on a CPU within 5 minutes.\n",
                "\n",
                "We will analyze:\n",
                "1. Data Loading & Flattening of ALL nested JSON properties.\n",
                "2. Missing Value Analysis.\n",
                "3. Distributions of features (Profile, Skills, Education, Signals).\n",
                "4. Fake Detection Heuristics (e.g., Logical contradictions, Keyword stuffing)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import json\n",
                "\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "plt.rcParams['figure.figsize'] = (12, 6)"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Data Loading and Flattening\n",
                "Since the dataset has ~100k records with deeply nested structures, we will parse the JSONL file and flatten it into a Pandas DataFrame for easier analysis."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "def flatten_candidate(c):\n",
                "    flat = {'candidate_id': c.get('candidate_id')}\n",
                "    \n",
                "    # Profile\n",
                "    profile = c.get('profile', {})\n",
                "    for k, v in profile.items():\n",
                "        flat[f'profile_{k}'] = v\n",
                "        \n",
                "    # Redrob Signals\n",
                "    signals = c.get('redrob_signals', {})\n",
                "    for k, v in signals.items():\n",
                "        if k == 'expected_salary_range_inr_lpa':\n",
                "            flat['expected_salary_min'] = v.get('min') if isinstance(v, dict) else None\n",
                "            flat['expected_salary_max'] = v.get('max') if isinstance(v, dict) else None\n",
                "        elif k == 'skill_assessment_scores':\n",
                "            # Store average assessment score or number of assessments taken\n",
                "            scores = list(v.values()) if isinstance(v, dict) else []\n",
                "            flat['signal_assessments_taken'] = len(scores)\n",
                "            flat['signal_avg_assessment_score'] = sum(scores) / len(scores) if scores else None\n",
                "        else:\n",
                "            flat[f'signal_{k}'] = v\n",
                "            \n",
                "    # Arrays (Career, Education, Skills, Certifications, Languages)\n",
                "    flat['num_jobs'] = len(c.get('career_history', []))\n",
                "    flat['num_degrees'] = len(c.get('education', []))\n",
                "    flat['num_skills'] = len(c.get('skills', []))\n",
                "    flat['num_certifications'] = len(c.get('certifications', []))\n",
                "    flat['num_languages'] = len(c.get('languages', []))\n",
                "    \n",
                "    # Skill specific aggregations\n",
                "    skills = c.get('skills', [])\n",
                "    flat['avg_skill_endorsements'] = sum(s.get('endorsements', 0) for s in skills) / len(skills) if skills else 0\n",
                "    flat['max_skill_duration_months'] = max([s.get('duration_months', 0) for s in skills], default=0)\n",
                "    \n",
                "    # Career specific aggregations\n",
                "    careers = c.get('career_history', [])\n",
                "    total_job_duration = sum(job.get('duration_months', 0) for job in careers)\n",
                "    flat['total_job_duration_months'] = total_job_duration\n",
                "    \n",
                "    return flat\n",
                "\n",
                "def load_data(filepath, sample_size=None):\n",
                "    records = []\n",
                "    with open(filepath, 'r') as f:\n",
                "        for i, line in enumerate(f):\n",
                "            if sample_size and i >= sample_size:\n",
                "                break\n",
                "            records.append(flatten_candidate(json.loads(line)))\n",
                "    return pd.DataFrame(records)\n",
                "\n",
                "# Load the first 10,000 rows for quick EDA, or set sample_size=None for all\n",
                "df = load_data('candidates.jsonl', sample_size=None)\n",
                "print(f\"Loaded {len(df)} records with {len(df.columns)} features.\")\n",
                "df.head()"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Missing Value Analysis\n",
                "Understanding what data is frequently missing is crucial. Missing signals might indicate incomplete profiles, which could be a factor in detecting low-quality or fake accounts."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "missing_perc = df.isnull().mean() * 100\n",
                "missing_perc = missing_perc[missing_perc > 0].sort_values(ascending=False)\n",
                "\n",
                "plt.figure(figsize=(10, 8))\n",
                "sns.barplot(x=missing_perc.values, y=missing_perc.index)\n",
                "plt.title('Percentage of Missing Values per Feature')\n",
                "plt.xlabel('% Missing')\n",
                "plt.show()"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Fake Detection Heuristics Analysis\n",
                "We will investigate several hypotheses that might indicate a profile is fake, keyword-stuffed, or heavily exaggerated."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Heuristic 1: Logical Contradictions (Experience vs. Job Durations)\n",
                "Does the `years_of_experience` declared in the profile match the sum of durations in their career history? Exaggerated profiles often claim high years of experience but have short job histories."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "df['declared_exp_months'] = df['profile_years_of_experience'] * 12\n",
                "df['exp_discrepancy'] = df['declared_exp_months'] - df['total_job_duration_months']\n",
                "\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.histplot(df['exp_discrepancy'], bins=50, kde=True)\n",
                "plt.title('Discrepancy: Declared Experience (months) - Total Job Duration (months)')\n",
                "plt.xlabel('Discrepancy (Months)')\n",
                "plt.ylabel('Count')\n",
                "plt.show()\n",
                "\n",
                "# Let's identify candidates with massive discrepancies (> 5 years off)\n",
                "high_discrepancy = df[df['exp_discrepancy'].abs() > 60]\n",
                "print(f\"Candidates with >5 years discrepancy between declared and actual experience: {len(high_discrepancy)} ({len(high_discrepancy)/len(df)*100:.2f}%)\")"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Heuristic 2: Keyword Stuffing (High Skills but Low Endorsements / Short Duration)\n",
                "Fake candidates often list 50+ skills but have 0 endorsements or claim to have used them for 0 months. Let's look at the relationship between number of skills and average endorsements."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "plt.figure(figsize=(10, 6))\n",
                "sns.scatterplot(x='num_skills', y='avg_skill_endorsements', data=df, alpha=0.5)\n",
                "plt.title('Number of Skills vs. Average Endorsements per Skill')\n",
                "plt.xlabel('Total Number of Skills Listed')\n",
                "plt.ylabel('Average Endorsements')\n",
                "plt.show()\n",
                "\n",
                "keyword_stuffers = df[(df['num_skills'] > 20) & (df['avg_skill_endorsements'] < 1)]\n",
                "print(f\"Potential Keyword Stuffers (>20 skills, <1 avg endorsement): {len(keyword_stuffers)} ({len(keyword_stuffers)/len(df)*100:.2f}%)\")"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Heuristic 3: Redrob Engagement Signals (Bot-like behavior)\n",
                "Bots or spam accounts might spam applications but have terrible recruiter response rates and profile completion scores."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
                "\n",
                "sns.scatterplot(x='signal_applications_submitted_30d', y='signal_recruiter_response_rate', data=df, alpha=0.5, ax=axes[0])\n",
                "axes[0].set_title('Applications vs Recruiter Response Rate')\n",
                "axes[0].set_xlabel('Applications Submitted (30 days)')\n",
                "axes[0].set_ylabel('Recruiter Response Rate')\n",
                "\n",
                "sns.histplot(df['signal_profile_completeness_score'], bins=20, kde=True, ax=axes[1])\n",
                "axes[1].set_title('Distribution of Profile Completeness Score')\n",
                "axes[1].set_xlabel('Completeness Score')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "spam_apps = df[(df['signal_applications_submitted_30d'] > 50) & (df['signal_recruiter_response_rate'] < 0.05)]\n",
                "print(f\"Spammers (>50 applications, <5% response rate): {len(spam_apps)} ({len(spam_apps)/len(df)*100:.2f}%)\")"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Heuristic 4: Unverified Accounts\n",
                "Does the account lack basic verification (Email, Phone, LinkedIn) while claiming top tier stats?"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "df['unverified_count'] = (~df['signal_verified_email']).astype(int) + \\\n",
                "                         (~df['signal_verified_phone']).astype(int) + \\\n",
                "                         (~df['signal_linkedin_connected']).astype(int)\n",
                "\n",
                "sns.countplot(x='unverified_count', data=df)\n",
                "plt.title('Number of Unverified Channels (Email, Phone, LinkedIn)')\n",
                "plt.xlabel('Count of Unverified Channels (0 = Fully Verified, 3 = Completely Unverified)')\n",
                "plt.show()\n",
                "\n",
                "completely_unverified = df[df['unverified_count'] == 3]\n",
                "print(f\"Completely Unverified Accounts: {len(completely_unverified)} ({len(completely_unverified)/len(df)*100:.2f}%)\")"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Conclusion & Next Steps for Stage 1 Filter\n",
                "Based on the analysis above, we can define a multi-factor **\"Trust Score\"** or a boolean **\"Is_Fake\"** filter.\n",
                "\n",
                "Proposed Rule-out conditions for Stage 1:\n",
                "1. **`exp_discrepancy`** > Threshold (e.g. 5 years).\n",
                "2. **`unverified_count`** == 3 (No email, phone, or LinkedIn).\n",
                "3. **`num_skills`** > Threshold AND **`avg_skill_endorsements`** < Threshold (Keyword stuffing).\n",
                "4. **`signal_profile_completeness_score`** < Minimum threshold.\n",
                "\n",
                "Candidates passing this lightweight CPU-bound check will proceed to Stage 2."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("EDA.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print("EDA.ipynb generated successfully!")
