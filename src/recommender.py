"""
Core Recommendation Engines: Content-Based, Collaborative Filtering (SVD), and Hybrid Recommender.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import joblib

from config import SCALER_PATH, SVD_PATH, NUM_SVD_COMPONENTS, RANDOM_STATE

class ContentBasedRecommender:
    """
    Content-Based Filtering engine using audio features & metadata cosine similarity.
    """
    def __init__(self):
        self.tracks_df: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None

    def fit(self, tracks_df: pd.DataFrame, feature_matrix: np.ndarray):
        self.tracks_df = tracks_df.copy().reset_index(drop=True)
        self.feature_matrix = feature_matrix
        self.similarity_matrix = cosine_similarity(feature_matrix)

    def recommend_similar_tracks(self, track_id: str, top_k: int = 10) -> pd.DataFrame:
        """
        Recommends tracks most similar to a given track_id.
        """
        if self.tracks_df is None or self.similarity_matrix is None:
            raise ValueError("Model must be fitted before generating recommendations.")

        idx_matches = self.tracks_df.index[self.tracks_df["track_id"] == track_id].tolist()
        if not idx_matches:
            raise ValueError(f"Track ID '{track_id}' not found in dataset.")

        track_idx = idx_matches[0]
        sim_scores = self.similarity_matrix[track_idx]

        # Sort indices by similarity score descending, excluding the query track itself
        sorted_indices = np.argsort(sim_scores)[::-1]
        recommended_indices = [i for i in sorted_indices if i != track_idx][:top_k]

        results = self.tracks_df.iloc[recommended_indices].copy()
        results["similarity_score"] = sim_scores[recommended_indices]
        return results

    def recommend_by_audio_profile(self, target_vector: np.ndarray, top_k: int = 10) -> pd.DataFrame:
        """
        Recommends tracks matching a target scaled audio feature vector.
        """
        sim_scores = cosine_similarity(target_vector.reshape(1, -1), self.feature_matrix)[0]
        sorted_indices = np.argsort(sim_scores)[::-1][:top_k]

        results = self.tracks_df.iloc[sorted_indices].copy()
        results["similarity_score"] = sim_scores[sorted_indices]
        return results


class CollaborativeRecommender:
    """
    Collaborative Filtering engine utilizing Matrix Factorization (TruncatedSVD).
    """
    def __init__(self, n_components: int = NUM_SVD_COMPONENTS, random_state: int = RANDOM_STATE):
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.pivot_df: Optional[pd.DataFrame] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.reconstructed_matrix: Optional[np.ndarray] = None

    def fit(self, pivot_df: pd.DataFrame):
        self.pivot_df = pivot_df
        # Fit SVD on User-Item Matrix
        num_comp = min(self.n_components, pivot_df.shape[1] - 1, pivot_df.shape[0] - 1)
        if num_comp < 1:
            num_comp = 1
        self.svd = TruncatedSVD(n_components=num_comp, random_state=self.random_state)
        
        self.user_factors = self.svd.fit_transform(pivot_df.values)
        self.item_factors = self.svd.components_
        self.reconstructed_matrix = np.dot(self.user_factors, self.item_factors)

    def predict_rating(self, user_id: str, track_id: str) -> float:
        if self.pivot_df is None:
            raise ValueError("Model is not fitted.")

        if user_id not in self.pivot_df.index or track_id not in self.pivot_df.columns:
            return 3.0 # Default global baseline rating

        user_idx = self.pivot_df.index.get_loc(user_id)
        track_idx = self.pivot_df.columns.get_loc(track_id)

        pred = self.reconstructed_matrix[user_idx, track_idx]
        return float(np.clip(pred, 1.0, 5.0))

    def recommend_for_user(self, user_id: str, tracks_df: pd.DataFrame, top_k: int = 10, filter_already_rated: bool = True) -> pd.DataFrame:
        if self.pivot_df is None:
            raise ValueError("Model is not fitted.")

        if user_id not in self.pivot_df.index:
            # Cold-start user: return popular tracks
            popular_tracks = tracks_df.sort_values(by="popularity", ascending=False).head(top_k).copy()
            popular_tracks["predicted_rating"] = 3.5
            return popular_tracks

        user_idx = self.pivot_df.index.get_loc(user_id)
        user_pred_scores = self.reconstructed_matrix[user_idx]
        track_ids = self.pivot_df.columns.tolist()

        pred_df = pd.DataFrame({
            "track_id": track_ids,
            "predicted_rating": user_pred_scores
        })

        if filter_already_rated:
            # Get tracks user has already rated (>0)
            user_ratings = self.pivot_df.loc[user_id]
            rated_tracks = user_ratings[user_ratings > 0].index.tolist()
            pred_df = pred_df[~pred_df["track_id"].isin(rated_tracks)]

        top_pred = pred_df.sort_values(by="predicted_rating", ascending=False).head(top_k)
        result = top_pred.merge(tracks_df, on="track_id", how="left")
        return result


class HybridRecommender:
    """
    Hybrid Recommender combining Content-Based and Collaborative Filtering predictions.
    """
    def __init__(self, content_model: ContentBasedRecommender, collab_model: CollaborativeRecommender):
        self.content_model = content_model
        self.collab_model = collab_model

    def recommend(
        self,
        user_id: str,
        tracks_df: pd.DataFrame,
        anchor_track_id: Optional[str] = None,
        alpha: float = 0.5,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Hybrid recommendation blending collaborative predictions and content similarities.
        alpha = weight for Collaborative Filtering (0.0 = 100% Content, 1.0 = 100% Collaborative).
        """
        df = tracks_df.copy()
        
        # 1. Collaborative Scores (Normalized 0..1)
        if self.collab_model.pivot_df is not None and user_id in self.collab_model.pivot_df.index:
            user_idx = self.collab_model.pivot_df.index.get_loc(user_id)
            track_ids = self.collab_model.pivot_df.columns.tolist()
            raw_cf_scores = self.collab_model.reconstructed_matrix[user_idx]
            
            # Map to tracks dataframe order
            cf_map = dict(zip(track_ids, raw_cf_scores))
            df["cf_raw"] = df["track_id"].map(cf_map).fillna(3.0)
            
            # Normalize 0..1
            min_c, max_c = df["cf_raw"].min(), df["cf_raw"].max()
            df["cf_score"] = (df["cf_raw"] - min_c) / (max_c - min_c + 1e-8)
        else:
            # Cold start user: fall back to track popularity normalized
            pop_min, pop_max = df["popularity"].min(), df["popularity"].max()
            df["cf_score"] = (df["popularity"] - pop_min) / (pop_max - pop_min + 1e-8)

        # 2. Content Scores (Normalized 0..1)
        if anchor_track_id and anchor_track_id in df["track_id"].values:
            anchor_idx = self.content_model.tracks_df.index[self.content_model.tracks_df["track_id"] == anchor_track_id][0]
            sim_scores = self.content_model.similarity_matrix[anchor_idx]
            df["content_score"] = sim_scores
        else:
            # Default content score based on overall popularity / audio features
            df["content_score"] = df["popularity"] / 100.0

        # 3. Hybrid Score Combination
        df["hybrid_score"] = alpha * df["cf_score"] + (1 - alpha) * df["content_score"]

        # Exclude anchor track if specified
        if anchor_track_id:
            df = df[df["track_id"] != anchor_track_id]

        recommendations = df.sort_values(by="hybrid_score", ascending=False).head(top_k).copy()
        return recommendations
