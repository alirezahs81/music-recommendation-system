"""
Data Preprocessing & Feature Engineering Module.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from config import AUDIO_FEATURES

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.feature_names = []

    def prepare_content_features(self, tracks_df: pd.DataFrame) -> np.ndarray:
        """
        Extracts, scales, and concatenates audio numerical features and genre/artist TF-IDF text features.
        """
        # 1. Scale numerical audio features
        audio_data = tracks_df[AUDIO_FEATURES].values
        scaled_audio = self.scaler.fit_transform(audio_data)

        # 2. Extract TF-IDF features from text metadata (genre + artist)
        text_metadata = (tracks_df["genre"].fillna("") + " " + tracks_df["artist"].fillna("")).tolist()
        tfidf_matrix = self.tfidf.fit_transform(text_metadata).toarray()

        # 3. Concatenate feature vectors
        combined_features = np.hstack([scaled_audio, tfidf_matrix])
        
        # Save feature names for inspectability
        text_feature_names = [f"tfidf_{name}" for name in self.tfidf.get_feature_names_out()]
        self.feature_names = AUDIO_FEATURES + text_feature_names

        return combined_features

    def build_user_item_matrix(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
        """
        Builds user-item interaction pivot table matrix.
        """
        pivot_df = ratings_df.pivot(index="user_id", columns="track_id", values="rating").fillna(0.0)
        
        user_to_idx = {user: idx for idx, user in enumerate(pivot_df.index)}
        track_to_idx = {track: idx for idx, track in enumerate(pivot_df.columns)}

        return pivot_df, user_to_idx, track_to_idx
