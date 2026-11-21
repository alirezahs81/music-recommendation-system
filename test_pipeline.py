"""Quick pipeline test script."""
import sys
sys.path.insert(0, r"C:\Users\Laptopkaran\Documents\Default Project\Music Recommendation System")

from src.data_loader import load_or_create_data
from src.preprocessing import DataPreprocessor
from src.recommender import ContentBasedRecommender, CollaborativeRecommender, HybridRecommender

tracks, ratings = load_or_create_data()
print(f"Tracks: {len(tracks)}, Ratings: {len(ratings)}")

preprocessor = DataPreprocessor()
feature_matrix = preprocessor.prepare_content_features(tracks)
print("Content features shape:", feature_matrix.shape)

cbf = ContentBasedRecommender()
cbf.fit(tracks, feature_matrix)
recs = cbf.recommend_similar_tracks("TRK0001", top_k=3)
print("Content-based recommendations OK")

pivot_df, user_to_idx, track_to_idx = preprocessor.build_user_item_matrix(ratings)
collab = CollaborativeRecommender(n_components=10)
collab.fit(pivot_df)
pred = collab.predict_rating("USR001", "TRK0001")
print(f"Collab prediction: {pred}")

hybrid = HybridRecommender(content_model=cbf, collab_model=collab)
hybrid_recs = hybrid.recommend("USR001", tracks, top_k=5)
print(f"Hybrid recommendations: {len(hybrid_recs)} items")

print("\n✅ Full pipeline test passed!")