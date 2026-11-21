"""
Data Loader and Synthetic Data Generator for Music Recommendation System.
"""
import numpy as np
import pandas as pd
from typing import Tuple
from config import TRACKS_FILE, RATINGS_FILE, GENRE_LIST, RANDOM_STATE

def generate_synthetic_music_dataset(
    num_tracks: int = 150,
    num_users: int = 50,
    num_ratings: int = 1200,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic synthetic track metadata and user rating data for recommendation benchmarking.
    """
    np.random.seed(random_state)

    # Track Titles & Artists components
    adjectives = ["Cosmic", "Neon", "Electric", "Silent", "Midnight", "Golden", "Velvet", "Dark", "Urban", "Melodic"]
    nouns = ["Echoes", "Dreams", "Rhythm", "Vibes", "Groove", "Symphony", "Whispers", "Flow", "Horizon", "Pulse"]
    artists = [f"Artist {chr(65 + i)}" for i in range(25)]

    track_records = []
    for i in range(1, num_tracks + 1):
        t_id = f"TRK{i:04d}"
        title = f"{np.random.choice(adjectives)} {np.random.choice(nouns)}"
        artist = np.random.choice(artists)
        genre = np.random.choice(GENRE_LIST)
        year = int(np.random.randint(2010, 2026))

        # Audio features (0.0 to 1.0 mostly, except tempo & loudness)
        danceability = round(float(np.random.uniform(0.1, 0.98)), 3)
        energy = round(float(np.random.uniform(0.1, 0.99)), 3)
        acousticness = round(float(np.random.uniform(0.01, 0.95)), 3)
        valence = round(float(np.random.uniform(0.05, 0.98)), 3)
        tempo = round(float(np.random.uniform(70.0, 175.0)), 1)
        loudness = round(float(np.random.uniform(-25.0, -2.0)), 1)
        popularity = int(np.random.randint(20, 100))

        track_records.append({
            "track_id": t_id,
            "title": title,
            "artist": artist,
            "genre": genre,
            "release_year": year,
            "danceability": danceability,
            "energy": energy,
            "acousticness": acousticness,
            "valence": valence,
            "tempo": tempo,
            "loudness": loudness,
            "popularity": popularity
        })

    tracks_df = pd.DataFrame(track_records)

    # Generate User Interactions / Ratings (1-5 star ratings or play counts)
    ratings_records = []
    user_ids = [f"USR{u:03d}" for u in range(1, num_users + 1)]
    track_ids = tracks_df["track_id"].tolist()

    # Give users preference bias towards certain genres to make collaborative filtering meaningful
    user_genre_prefs = {u: np.random.choice(GENRE_LIST, size=2, replace=False) for u in user_ids}

    interactions_count = 0
    while interactions_count < num_ratings:
        u = np.random.choice(user_ids)
        t_row = tracks_df.sample(1).iloc[0]
        t = t_row["track_id"]
        t_genre = t_row["genre"]

        # Higher rating if genre matches user preference
        if t_genre in user_genre_prefs[u]:
            rating = float(np.random.choice([4.0, 5.0, 5.0], p=[0.2, 0.4, 0.4]))
        else:
            rating = float(np.random.choice([1.0, 2.0, 3.0, 4.0], p=[0.2, 0.3, 0.3, 0.2]))

        ratings_records.append({
            "user_id": u,
            "track_id": t,
            "rating": rating
        })
        interactions_count += 1

    ratings_df = pd.DataFrame(ratings_records).drop_duplicates(subset=["user_id", "track_id"])

    return tracks_df, ratings_df


def load_or_create_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads tracks and ratings dataset from disk, or creates synthetic datasets if not present.
    """
    if TRACKS_FILE.exists() and RATINGS_FILE.exists():
        tracks_df = pd.read_csv(TRACKS_FILE)
        ratings_df = pd.read_csv(RATINGS_FILE)
    else:
        tracks_df, ratings_df = generate_synthetic_music_dataset()
        tracks_df.to_csv(TRACKS_FILE, index=False)
        ratings_df.to_csv(RATINGS_FILE, index=False)

    return tracks_df, ratings_df
