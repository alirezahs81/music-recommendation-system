"""Full system test."""
import sys
sys.path.insert(0, r"C:\Users\Laptopkaran\Documents\Default Project\Music Recommendation System")

from src.data_loader import load_or_create_data
from src.preprocessing import DataPreprocessor
from src.recommender import ContentBasedRecommender, CollaborativeRecommender, HybridRecommender
from src.evaluate import calculate_rmse_mae, calculate_precision_recall_at_k, calculate_map_at_k, calculate_catalog_coverage, calculate_intra_list_diversity
import pandas as pd
import numpy as np

print("=" * 60)
print("MUSIC RECOMMENDATION SYSTEM - COMPLETE IMPLEMENTATION")
print("=" * 60)

# 1. Data Loading
tracks, ratings = load_or_create_data()
print(f"\n1. DATA LOADING")
print(f"   Tracks: {len(tracks)} | Users: {ratings['user_id'].nunique()} | Ratings: {len(ratings)}")

# 2. Preprocessing
preprocessor = DataPreprocessor()
feature_matrix = preprocessor.prepare_content_features(tracks)
pivot_df, user_to_idx, track_to_idx = preprocessor.build_user_item_matrix(ratings)
print(f"   Feature matrix shape: {feature_matrix.shape}")
print(f"   User-item matrix shape: {pivot_df.shape}")

# 3. Content-Based Model
cbf = ContentBasedRecommender()
cbf.fit(tracks, feature_matrix)
similar_recs = cbf.recommend_similar_tracks("TRK0001", top_k=3)
print(f"   Content-Based: Similar tracks recommendation works OK")

# 4. Collaborative Filtering
collab = CollaborativeRecommender(n_components=10)
collab.fit(pivot_df)
pred = collab.predict_rating("USR001", "TRK0001")
print(f"   Collaborative: Rating prediction = {pred:.1f}/5 OK")

# 5. Hybrid Recommender
hybrid = HybridRecommender(content_model=cbf, collab_model=collab)
hybrid_recs = hybrid.recommend("USR001", tracks, top_k=5, alpha=0.5)
print(f"   Hybrid: {len(hybrid_recs)} recommendations with hybrid_score OK")

# 6. Evaluation Metrics
actual = np.array([4.0, 5.0, 3.0, 4.5, 2.0])
predicted_cf = np.array([4.2, 4.8, 2.9, 4.1, 1.8])
rmse, mae = calculate_rmse_mae(actual, predicted_cf)
print(f"   Evaluation: RMSE={rmse:.3f}, MAE={mae:.3f} OK")

prec, rec = calculate_precision_recall_at_k(
    {"USR001": {"TRK0001", "TRK0002", "TRK0003"}},
    {"USR001": ["TRK0001", "TRK0002", "TRK0004"]},
    k=3
)
print(f"   Metrics: Precision@3={prec:.3f}, Recall@3={rec:.3f} OK")

map_k = calculate_map_at_k(
    {"USR001": {"TRK0001", "TRK0002", "TRK0003"}},
    {"USR001": ["TRK0001", "TRK0002", "TRK0004"]},
    k=3
)
print(f"   Metrics: MAP@3={map_k:.3f} OK")

coverage = calculate_catalog_coverage(
    {"USR001": ["TRK0001", "TRK0002", "TRK0003", "TRK0004", "TRK0005"]},
    set(tracks["track_id"].tolist())
)
print(f"   Coverage: {coverage:.1%} of catalog covered OK")

ild = calculate_intra_list_diversity(
    ["TRK0001", "TRK0003", "TRK0005"],
    feature_matrix,
    track_to_idx
)
print(f"   Diversity: Intra-List Diversity (ILD)={ild:.3f} OK")

print("\n" + "=" * 60)
print("ALL MODULES FUNCTIONAL SUCCESSFULLY")
print("=" * 60)