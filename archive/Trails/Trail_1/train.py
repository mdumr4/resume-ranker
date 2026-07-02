import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from pathlib import Path
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LambdaMARTTrainer:
    """
    Trains the XGBoost LambdaMART model for learning the feature weights and interactions.
    Requires pseudo-labeled data. For a true JD-agnostic system, we would generate pseudo-labels 
    for multiple JDs and concatenate the training data.
    """
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "lambdamart.json"

    def prepare_training_data(self, features_df: pd.DataFrame, num_jds=3) -> Tuple[xgb.DMatrix, np.ndarray]:
        """
        Prepares training data for LambdaMART. 
        In production, this is where we call the LLM to pseudo-label a stratified sample 
        of candidates against a set of JDs.
        """
        logging.info("Generating pseudo-labels for LambdaMART training (Mocking for now)...")
        
        # We need a simulated dataset of cross-encoder scores + tabular features + relevance label
        # Features that LambdaMART will see:
        # 1. ce_career_fit
        # 2. ce_skills_fit
        # 3. ce_profile_fit
        # 4. ce_edu_fit
        # 5. Tabular features (trust, behavioral, structural)
        
        # Select the features we want LambdaMART to train on from the pre-computed dataframe
        tabular_cols = [
            'years_of_experience', 'is_india_based', 'profile_completeness',
            'open_to_work', 'response_rate', 'notice_period_days', 'github_score',
            'interview_completion', 'trust_exp_mismatch_months', 'trust_date_violation',
            'trust_salary_violation', 'trust_unverified', 'trust_expert_stuffing_risk',
            'avg_skill_proficiency', 'avg_skill_duration_months'
        ]
        
        # Filter only existing columns
        tabular_cols = [c for c in tabular_cols if c in features_df.columns]
        
        # Sample 1000 candidates for training
        n_samples = min(1000, len(features_df))
        train_df = features_df.sample(n=n_samples, random_state=42).copy()
        X_tabular = train_df[tabular_cols].values
        
        # Simulate Cross-Encoder section scores (since we don't have them pre-computed, they happen at runtime)
        # We simulate them to train the model how to weight them.
        X_ce = np.random.uniform(0.1, 0.9, size=(n_samples, 4))
        
        # Combine all features for XGBoost
        X = np.hstack([X_ce, X_tabular])
        
        # Simulate LLM labels (0 to 5 relevance score).
        # We generate it such that high CE scores + high trust = high label
        # to teach the model the interaction.
        labels = np.zeros(n_samples)
        
        ce_mean = X_ce.mean(axis=1) # average of simulated cross-encoder scores
        
        trust_penalty = np.zeros(n_samples)
        if 'trust_date_violation' in tabular_cols:
            idx = tabular_cols.index('trust_date_violation')
            trust_penalty += X_tabular[:, idx] * 2.0
            
        if 'trust_expert_stuffing_risk' in tabular_cols:
            idx = tabular_cols.index('trust_expert_stuffing_risk')
            trust_penalty += X_tabular[:, idx] * 1.5
            
        simulated_scores = (ce_mean * 5) - trust_penalty
        labels = np.clip(np.round(simulated_scores), 0, 5)
        
        # LambdaMART requires data to be grouped by query (JD).
        # We simulate training across 3 JDs.
        group_sizes = [n_samples // num_jds] * num_jds
        group_sizes[-1] += n_samples - sum(group_sizes) # Add remainder to last group
        
        dtrain = xgb.DMatrix(X, label=labels)
        dtrain.set_group(group_sizes)
        
        # Store feature names for later interpretation
        feature_names = ['ce_career', 'ce_skills', 'ce_profile', 'ce_edu'] + tabular_cols
        
        return dtrain, feature_names

    def train(self, features_parquet: str = "index/features.parquet"):
        if not Path(features_parquet).exists():
            logging.error(f"Cannot find {features_parquet}. Run index_builder first.")
            return
            
        df = pd.read_parquet(features_parquet)
        dtrain, feature_names = self.prepare_training_data(df)
        
        # LambdaMART parameters
        params = {
            "objective": "rank:ndcg",
            "eval_metric": ["ndcg@10", "ndcg@50"],
            "eta": 0.1,
            "max_depth": 5,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        
        logging.info("Training LambdaMART model...")
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=100, 
            evals=[(dtrain, "train")], 
            verbose_eval=10
        )
        
        model.feature_names = feature_names
        model.save_model(self.model_path)
        logging.info(f"Model saved to {self.model_path}")
        
        # Print feature importances
        importance = model.get_score(importance_type='gain')
        logging.info("Top Feature Importances (Gain):")
        for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            logging.info(f"  {k}: {v:.2f}")

if __name__ == "__main__":
    from typing import Tuple
    trainer = LambdaMARTTrainer(model_dir="models")
    trainer.train(features_parquet="index/features.parquet")
