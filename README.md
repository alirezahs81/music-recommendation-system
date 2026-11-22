# 🎵 Music Recommendation System

A hybrid music recommendation system built with Streamlit, combining **Content-Based Filtering** (audio features + TF-IDF metadata with cosine similarity) and **Collaborative Filtering** (TruncatedSVD matrix factorization).

## Features
- 📊 **Data Overview** — dataset stats, genre distribution, popularity histogram
- 🔍 **Recommendations** — Content-Based (similar tracks), Collaborative Filtering, and Hybrid mode with adjustable α weight and optional anchor track
- 📈 **Evaluation** — RMSE/MAE, Precision@K, Recall@K, MAP@K, Catalog Coverage, Intra-List Diversity (train/test split based)
- ⚙️ **Settings** — dataset info, feature selection, model hyperparameters

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
app.py                  # Streamlit UI
config.py               # Paths & hyperparameters
src/
  data_loader.py        # Data loading + synthetic dataset generator
  preprocessing.py      # Feature scaling, TF-IDF, user-item matrix
  recommender.py        # Content-Based, Collaborative (SVD), Hybrid engines
  evaluate.py           # Evaluation metrics
data/                   # tracks.csv, user_ratings.csv (auto-generated)
models/                 # Saved model artifacts
```

## Notes
If `data/` CSVs are missing, a realistic synthetic dataset (150 tracks, 50 users, ~1200 ratings) is generated automatically on first run.
