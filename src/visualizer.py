"""
Data Visualization and Chart Generator for Music Recommendations.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional
from config import AUDIO_FEATURES

def plot_genre_distribution(tracks_df: pd.DataFrame) -> plt.Figure:
    """
    Plots Matplotlib bar chart of track count per genre.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    genre_counts = tracks_df["genre"].value_counts()
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="viridis", ax=ax)
    ax.set_title("Track Distribution by Genre", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Tracks")
    ax.set_ylabel("Genre")
    plt.tight_layout()
    return fig

def plot_audio_features_correlation(tracks_df: pd.DataFrame) -> plt.Figure:
    """
    Plots correlation heatmap of numerical audio features.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = tracks_df[AUDIO_FEATURES].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=True)
    ax.set_title("Audio Features Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig

def create_audio_radar_chart(track_rows: pd.DataFrame, title: str = "Audio Feature Profile Comparison") -> go.Figure:
    """
    Generates interactive Plotly Radar Chart comparing audio profiles of selected tracks.
    """
    radar_cols = ["danceability", "energy", "acousticness", "valence"]
    
    fig = go.Figure()
    
    for idx, row in track_rows.iterrows():
        values = [row[col] for col in radar_cols]
        values.append(values[0]) # close loop
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_cols + [radar_cols[0]],
            fill='toself',
            name=f"{row['title']} - {row['artist']}"
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title=title,
        template="plotly_dark"
    )
    return fig

def create_recommendation_bar_chart(recommendations_df: pd.DataFrame, score_col: str = "hybrid_score") -> go.Figure:
    """
    Creates Plotly Horizontal Bar Chart of top recommended tracks and their scores.
    """
    df = recommendations_df.sort_values(by=score_col, ascending=True).copy()
    df["label"] = df["title"] + " (" + df["artist"] + ")"
    
    fig = px.bar(
        df,
        x=score_col,
        y="label",
        orientation="h",
        color=score_col,
        color_continuous_scale="Viridis",
        title="Top Recommended Tracks & Recommendation Scores",
        labels={"label": "Track", score_col: "Score"}
    )
    fig.update_layout(template="plotly_dark", height=400)
    return fig
