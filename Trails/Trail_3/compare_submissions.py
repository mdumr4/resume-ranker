import pandas as pd

def compare_submissions():
    # Load both submissions
    t2_df = pd.read_csv("Trails/Trail_2/submission.csv")
    t3_df = pd.read_csv("Trails/Trail_3/submission.csv")
    
    print("=== Submissions Comparison (Trail 2 vs Trail 3) ===")
    
    # Overlap Analysis
    t2_top_10 = set(t2_df.head(10)['candidate_id'])
    t3_top_10 = set(t3_df.head(10)['candidate_id'])
    
    overlap_10 = len(t2_top_10.intersection(t3_top_10))
    print(f"Top 10 Overlap: {overlap_10} / 10")
    
    t2_top_50 = set(t2_df.head(50)['candidate_id'])
    t3_top_50 = set(t3_df.head(50)['candidate_id'])
    overlap_50 = len(t2_top_50.intersection(t3_top_50))
    print(f"Top 50 Overlap: {overlap_50} / 50")
    
    print("\n--- Top 5 Differences ---")
    print("Trail 2 (L-6-v2) Top 5:")
    for _, row in t2_df.head(5).iterrows():
        print(f"  Rank {row['rank']}: {row['candidate_id']} | Score: {row['score']:.4f}")
        
    print("\nTrail 3 (L-12-v2) Top 5:")
    for _, row in t3_df.head(5).iterrows():
        print(f"  Rank {row['rank']}: {row['candidate_id']} | Score: {row['score']:.4f}")

    print("\n--- Score Distribution Comparison ---")
    print(f"Trail 2 - Mean: {t2_df['score'].mean():.4f}, Max: {t2_df['score'].max():.4f}, Min: {t2_df['score'].min():.4f}")
    print(f"Trail 3 - Mean: {t3_df['score'].mean():.4f}, Max: {t3_df['score'].max():.4f}, Min: {t3_df['score'].min():.4f}")
    
if __name__ == "__main__":
    compare_submissions()
