"""
Configuration module for Music Recommendation System.
"""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

TRACKS_FILE = DATA_DIR / "tracks.csv"
RATINGS_FILE = DATA_DIR / "user_ratings.csv"

# Model Artifact Paths
SCALER_PATH = MODELS_DIR / "content_scaler.joblib"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
SVD_PATH = MODELS_DIR / "collaborative_svd.joblib"

# Model Hyperparameters
NUM_SVD_COMPONENTS = 10
RANDOM_STATE = 42

# Default Recommendations Count
DEFAULT_TOP_K = 10

# Audio Feature Columns
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "acousticness",
    "valence",
    "tempo",
    "loudness",
    "popularity"
]

GENRE_LIST = [
    "Pop", "Rock", "Hip-Hop", "Electronic", "Jazz",
    "Classical", "R&B", "Indie", "Metal", "Acoustic"
]
