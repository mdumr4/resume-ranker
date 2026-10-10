import csv
import re
from pathlib import Path
from typing import List, Union
import pandas as pd

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")
EXPECTED_DATA_ROWS = 100


def validate_submission_df(df: pd.DataFrame) -> List[str]:
    """
    Validates a candidate ranking DataFrame according to the challenge rules:
    1. Exactly 100 rows.
    2. Required columns: candidate_id, rank, score, reasoning.
    3. candidate_id matches CAND_XXXXXXX pattern, no duplicates.
    4. rank is unique integer from 1 to 100.
    5. score is float, monotonically non-increasing by rank.
    6. Equal scores tie-break requires candidate_id ascending.
    7. reasoning is non-empty string.
    """
    errors = []

    # Check columns
    actual_cols = list(df.columns)
    if actual_cols != REQUIRED_HEADER:
        errors.append(f"Header mismatch. Expected {REQUIRED_HEADER}, found {actual_cols}")
        return errors

    if len(df) != EXPECTED_DATA_ROWS:
        errors.append(f"Expected exactly {EXPECTED_DATA_ROWS} data rows, found {len(df)}")

    seen_ids = set()
    seen_ranks = set()
    by_rank = []

    for i, row in df.iterrows():
        row_num = i + 2
        cid = str(row["candidate_id"]).strip()
        rank_val = row["rank"]
        score_val = row["score"]
        reasoning_val = str(row.get("reasoning", "")).strip()

        # Check candidate_id
        if not cid:
            errors.append(f"Row {row_num}: candidate_id is empty.")
        elif not CANDIDATE_ID_PATTERN.match(cid):
            errors.append(f"Row {row_num}: candidate_id '{cid}' invalid format (must be CAND_XXXXXXX).")
        elif cid in seen_ids:
            errors.append(f"Row {row_num}: duplicate candidate_id '{cid}'.")
        else:
            seen_ids.add(cid)

        # Check rank
        try:
            rank = int(rank_val)
            if not 1 <= rank <= 100:
                errors.append(f"Row {row_num}: rank must be 1-100, got {rank}.")
            elif rank in seen_ranks:
                errors.append(f"Row {row_num}: duplicate rank {rank}.")
            else:
                seen_ranks.add(rank)
        except (ValueError, TypeError):
            errors.append(f"Row {row_num}: rank '{rank_val}' is not a valid integer.")
            rank = None

        # Check score
        try:
            score = float(score_val)
        except (ValueError, TypeError):
            errors.append(f"Row {row_num}: score '{score_val}' is not a valid float.")
            score = None

        # Check reasoning
        if not reasoning_val or reasoning_val.lower() == "nan":
            errors.append(f"Row {row_num}: reasoning must not be empty.")

        if rank is not None and score is not None and cid:
            by_rank.append((rank, score, cid))

    missing = set(range(1, 101)) - seen_ranks
    if missing:
        errors.append(f"Each rank 1-100 must appear once. Missing: {sorted(missing)}")

    by_rank.sort(key=lambda x: x[0])

    # Monotonicity check
    for idx in range(len(by_rank) - 1):
        r1, s1, c1 = by_rank[idx]
        r2, s2, c2 = by_rank[idx + 1]
        if s1 < s2:
            errors.append(f"Score must be non-increasing by rank: rank {r1} ({s1}) < rank {r2} ({s2})")
        elif s1 == s2 and c1 > c2:
            errors.append(f"Equal scores tie-break failed: rank {r1} ({c1}) > rank {r2} ({c2})")

    return errors


def validate_submission_file(csv_path: Union[str, Path]) -> List[str]:
    """Validates a CSV submission file on disk."""
    path = Path(csv_path)
    if not path.exists():
        return [f"File does not exist: {path}"]
    if path.suffix.lower() != ".csv":
        return ["File must use a .csv extension."]

    try:
        df = pd.read_csv(path)
        return validate_submission_df(df)
    except Exception as e:
        return [f"Failed to read CSV: {e}"]
