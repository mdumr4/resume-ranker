import pandas as pd
import argparse

def compare_trails(trail_a_csv, trail_b_csv):
    """
    Compares the final Cross-Encoder rankings between two different Trails.
    Assumes CSVs have columns: ['candidate_id', 'rank', 'score']
    """
    print(f"Loading {trail_a_csv} and {trail_b_csv}...\n")
    try:
        df_a = pd.read_csv(trail_a_csv)
        df_b = pd.read_csv(trail_b_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you save your Trail outputs to CSV files with columns ['candidate_id', 'rank', 'score']")
        return

    # Ensure sorted by rank
    df_a = df_a.sort_values('rank').reset_index(drop=True)
    df_b = df_b.sort_values('rank').reset_index(drop=True)

    print("========================================")
    print("🏆 ALL 100 CANDIDATES COMPARISON 🏆")
    print("========================================")
    
    top_n = min(100, len(df_a), len(df_b))
    print(f"{'Rank':<5} | {'Trail A (' + trail_a_csv + ')':<30} | {'Trail B (' + trail_b_csv + ')':<30}")
    print("-" * 75)
    for i in range(top_n):
        cand_a = df_a.iloc[i]['candidate_id'] if i < len(df_a) else "N/A"
        cand_b = df_b.iloc[i]['candidate_id'] if i < len(df_b) else "N/A"
        print(f"{i+1:<5} | {cand_a:<30} | {cand_b:<30}")

    print("\n========================================")
    print("🔄 FULL RANK SHIFT ANALYSIS 🔄")
    print("========================================")
    
    # Merge on candidate ID to see how ranks shifted
    merged = pd.merge(df_a, df_b, on='candidate_id', suffixes=('_A', '_B'))
    merged['rank_shift'] = merged['rank_A'] - merged['rank_B']  # Positive means they went UP in rank in Trail B
    
    biggest_winners = merged.sort_values('rank_shift', ascending=False)
    biggest_losers = merged.sort_values('rank_shift', ascending=True)
    
    print("Top 15 Winners (Boosted the most in Trail B):")
    for _, row in biggest_winners.head(15).iterrows():
        print(f"  - {row['candidate_id']}: Rank {row['rank_A']} -> Rank {row['rank_B']} (+{row['rank_shift']} spots)")
        
    print("\nTop 15 Losers (Penalized the most in Trail B):")
    for _, row in biggest_losers.head(15).iterrows():
        print(f"  - {row['candidate_id']}: Rank {row['rank_A']} -> Rank {row['rank_B']} ({row['rank_shift']} spots)")
        
    # Save full analysis to CSV
    output_file = "rank_shift_analysis.csv"
    merged.sort_values('rank_B').to_csv(output_file, index=False)
    print(f"\n✅ Full overlap and rank shift analysis for all candidates saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two Trail outputs.")
    parser.add_argument("trail_a", help="Path to first CSV")
    parser.add_argument("trail_b", help="Path to second CSV")
    args = parser.parse_args()
    
    compare_trails(args.trail_a, args.trail_b)
