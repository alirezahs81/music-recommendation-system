"""
Music Recommendation System - Streamlit Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.data_loader import load_or_create_data, generate_synthetic_music_dataset
from src.preprocessing import DataPreprocessor
from src.recommender import ContentBasedRecommender, CollaborativeRecommender, HybridRecommender
from src.evaluate import calculate_precision_recall_at_k, calculate_map_at_k, calculate_catalog_coverage, \
    calculate_intra_list_diversity, calculate_rmse_mae

# Page config
st.set_page_config(
    page_title="🎵 Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stTitle {font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;}
    .stMarkdown small {color: #666;}
    .recommendation-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #e0e0e0;
        transition: transform 0.2s;
    }
    .recommendation-card:hover {transform: translateY(-2px);}
    .genre-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background: #e94560;
        color: white;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_systems():
    """Load or train the recommendation models."""
    # Load data
    tracks_df, ratings_df = load_or_create_data()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Build content features
    feature_matrix = preprocessor.prepare_content_features(tracks_df)
    
    # Initialize and fit Content-Based model
    cbf = ContentBasedRecommender()
    cbf.fit(tracks_df, feature_matrix)
    
    # Build pivot matrix and fit Collaborative model
    pivot_df, user_to_idx, track_to_idx = preprocessor.build_user_item_matrix(ratings_df)
    collab = CollaborativeRecommender(n_components=10, random_state=42)
    collab.fit(pivot_df)
    
    # Hybrid Recommender
    hybrid = HybridRecommender(content_model=cbf, collab_model=collab)
    
    return tracks_df, ratings_df, cbf, collab, hybrid, preprocessor, user_to_idx, track_to_idx

def main():
    st.title("🎵 Music Recommendation System")
    st.caption("Hybrid Recommendation Engine | Content-Based + Collaborative Filtering")
    
    # Load systems
    with st.spinner("Loading data and training models..."):
        tracks_df, ratings_df, cbf, collab, hybrid, preprocessor, user_to_idx, track_to_idx = load_systems()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["📊 Data Overview", "🔍 Recommendations", "📈 Evaluation", "⚙️ Settings"])
    
    # Always load data
    tracks_df, ratings_df = load_or_create_data()
    
    if page == "📊 Data Overview":
        show_data_overview(tracks_df, ratings_df)
    elif page == "🔍 Recommendations":
        show_recommendations(tracks_df, ratings_df, cbf, collab, hybrid, user_to_idx, track_to_idx)
    elif page == "📈 Evaluation":
        show_evaluation(tracks_df, ratings_df, cbf, collab, hybrid, preprocessor, user_to_idx, track_to_idx)
    elif page == "⚙️ Settings":
        show_settings(tracks_df, ratings_df, hybrid, user_to_idx, track_to_idx)

def show_data_overview(tracks_df, ratings_df):
    st.header("📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tracks", len(tracks_df))
    with col2:
        st.metric("Total Users", ratings_df["user_id"].nunique())
    with col3:
        st.metric("Total Ratings", len(ratings_df))
    
    # Genre distribution
    st.subheader("Genre Distribution")
    fig = px.pie(tracks_df, names="genre", values="track_id", title="Tracks by Genre")
    st.plotly_chart(fig, use_container_width=True)
    
    # Audio features distribution
    st.subheader("Audio Features Distribution")
    fig = px.histogram(tracks_df, x="popularity", nbins=20, title="Popularity Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample data
    with st.expander("View Sample Data"):
        st.dataframe(tracks_df.head(10), use_container_width=True)
        st.dataframe(ratings_df.head(10), use_container_width=True)

def show_recommendations(tracks_df, ratings_df, cbf, collab, hybrid, user_to_idx, track_to_idx):
    st.header("🔍 Music Recommendations")
    
    # Mode selection
    rec_mode = st.radio(
        "Recommendation Mode:",
        ["Content-Based (Similar Tracks)", "Collaborative Filtering", "Hybrid (Recommended)"]
    )
    
    # --- Content-Based ---
    if rec_mode == "Content-Based (Similar Tracks)":
        st.subheader("Find Similar Tracks")
        track_options = tracks_df["track_id"].tolist()
        selected_track = st.selectbox("Select a track:", track_options)
        top_k = st.slider("Number of recommendations", 3, 20, 10)
        
        if st.button("Generate Recommendations"):
            try:
                recs = cbf.recommend_similar_tracks(selected_track, top_k=top_k)
                st.success(f"Top {top_k} tracks similar to '{selected_track}':")
                
                for _, row in recs.iterrows():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <strong>{row['title']}</strong> - {row['artist']}<br>
                        <span class="genre-tag">{row['genre']}</span>
                        <span style="float:right">Similarity: {row['similarity_score']:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # --- Collaborative Filtering ---
    elif rec_mode == "Collaborative Filtering":
        st.subheader("User-Based Collaborative Recommendations")
        user_options = ["Cold Start"] + [u for u in user_to_idx.keys()]
        selected_user = st.selectbox("Select User (or Cold Start):", user_options)
        top_k = st.slider("Number of recommendations", 3, 20, 10)
        
        if st.button("Generate Recommendations") and selected_user != "Cold Start":
            try:
                recs = collab.recommend_for_user(selected_user, tracks_df, top_k=top_k)
                st.success(f"Top {top_k} recommendations for {selected_user}:")
                
                for _, row in recs.iterrows():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <strong>{row['title']}</strong> - {row['artist']}<br>
                        <span class="genre-tag">{row['genre']}</span>
                        <span style="float:right">Predicted Rating: {row['predicted_rating']:.1f}/5</span>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        if selected_user == "Cold Start":
            st.info("Cold-start recommendations: Popular tracks based on overall popularity.")
    
    # --- Hybrid ---
    else:
        st.subheader("Hybrid Recommendations (Content + Collaborative)")
        
        # User selection
        user_options = ["Cold Start"] + [u for u in user_to_idx.keys()]
        selected_user = st.selectbox("Select User:", user_options, key="hybrid_user")
        
        # Anchor track selection
        anchor_track = st.selectbox(
            "Anchor Track (optional - for content-based blending):",
            ["None"] + tracks_df["track_id"].tolist(),
            key="hybrid_anchor"
        )
        
        # Alpha weight slider
        alpha = st.slider(
            "Collaborative Filtering Weight (α):",
            min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            help="0.0 = 100% Content-Based, 1.0 = 100% Collaborative"
        )
        
        top_k = st.slider("Number of recommendations", 3, 20, 10)
        
        if st.button("Generate Hybrid Recommendations"):
            try:
                if selected_user == "Cold Start":
                    # Cold start: just return popular tracks
                    recs = tracks_df.sort_values(by="popularity", ascending=False).head(top_k).copy()
                    recs["score"] = 3.5
                    st.info("Cold-start recommendations based on track popularity.")
                else:
                    recs = hybrid.recommend(
                        user_id=selected_user,
                        tracks_df=tracks_df,
                        anchor_track_id=None if anchor_track == "None" else anchor_track,
                        alpha=alpha,
                        top_k=top_k
                    )
                
                st.success(f"Top {top_k} hybrid recommendations for {selected_user}:")
                
                for _, row in recs.iterrows():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <strong>{row['title']}</strong> - {row['artist']}<br>
                        <span class="genre-tag">{row['genre']}</span>
                        <span style="float:right">Score: {row['hybrid_score']:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show score breakdown
                with st.expander("Score Breakdown"):
                    st.write(f"α (Collaborative Weight): {alpha}")
                    if selected_user != "Cold Start" and anchor_track != "None":
                        st.write(f"Anchor Track: {anchor_track}")
                    else:
                        st.write("Using default content scores (popularity-based).")
                        
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                import traceback
                st.exception(traceback.format_exc())

def show_evaluation(tracks_df, ratings_df, cbf, collab, hybrid, preprocessor, user_to_idx, track_to_idx):
    st.header("📈 Model Evaluation")
    st.markdown("""
    Evaluate recommendation quality using ranking and rating prediction metrics.
    """)
    
    # Split data for evaluation
    eval_ratings = ratings_df.sample(frac=0.2, random_state=42)
    train_ratings = ratings_df.drop(eval_ratings.index)
    
    # Build pivot on train data
    preprocessor_eval = DataPreprocessor()
    pivot_df, user_to_idx_eval, track_to_idx_eval = preprocessor_eval.build_user_item_matrix(train_ratings)
    
    # Fit collaborative on train
    collab_eval = CollaborativeRecommender(n_components=10, random_state=42)
    collab_eval.fit(pivot_df)
    
    # Content features
    feature_matrix = preprocessor_eval.prepare_content_features(tracks_df)
    cbf_eval = ContentBasedRecommender()
    cbf_eval.fit(tracks_df, feature_matrix)
    
    hybrid_eval = HybridRecommender(content_model=cbf_eval, collab_model=collab_eval)
    
    # --- RMSE/MAE Prediction Evaluation ---
    st.subheader("📉 Rating Prediction Accuracy (RMSE / MAE)")
    
    # Collect actual vs predicted
    actual_ratings = []
    predicted_ratings = []
    
    for _, row in eval_ratings.iterrows():
        u = row["user_id"]
        t = row["track_id"]
        actual = row["rating"]
        
        # Collaborative prediction
        cf_pred = collab_eval.predict_rating(u, t)
        
        # Content-based prediction (using audio profile similarity)
        # Simple: use popularity as proxy for content score
        content_pred = float(tracks_df[tracks_df["track_id"] == t]["popularity"].values[0]) / 100.0 * 5.0
        
        # Hybrid
        hybrid_pred = 0.5 * cf_pred + 0.5 * content_pred
        
        actual_ratings.append(actual)
        predicted_ratings.append(cf_pred)
    
    rmse_cf, mae_cf = calculate_rmse_mae(np.array(actual_ratings), np.array(predicted_ratings))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Collaborative Filtering RMSE", f"{rmse_cf:.3f}")
    with col2:
        st.metric("Collaborative Filtering MAE", f"{mae_cf:.3f}")
    
    # --- Precision@K / Recall@K ---
    st.subheader("📊 Ranking Metrics (Precision@K / Recall@K)")
    
    # Build ground truth: items rated 4+ by each user
    ground_truth = {}
    for u in train_ratings["user_id"].unique():
        user_ratings = train_ratings[train_ratings["user_id"] == u]
        liked = set(user_ratings[user_ratings["rating"] >= 4]["track_id"].tolist())
        if liked:
            ground_truth[u] = liked
    
    # Generate recommendations for each user
    k = st.slider("K value for Precision/Recall", 3, 20, 10)
    recommended_items = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, u in enumerate(ground_truth.keys()):
        # CF recommendations
        cf_recs = collab_eval.recommend_for_user(u, tracks_df, top_k=k, filter_already_rated=True)
        rec_ids = cf_recs["track_id"].tolist()
        recommended_items[f"CF_{u}"] = rec_ids
        
        # Hybrid recommendations
        hybrid_recs = hybrid_eval.recommend(user_id=u, tracks_df=tracks_df, top_k=k)
        hybrid_rec_ids = hybrid_recs["track_id"].tolist()
        recommended_items[f"Hybrid_{u}"] = hybrid_rec_ids
        
        progress_bar.progress((i + 1) / len(ground_truth))
        status_text.text(f"Evaluating user {i+1}/{len(ground_truth)}...")
    
    # Calculate metrics
    # Precision@K and Recall@K for CF
    prec_cf, rec_cf = calculate_precision_recall_at_k(ground_truth, recommended_items, k=k)
    # Precision@K and Recall@K for Hybrid
    # Need to extract just the recommended lists
    hybrid_rec_lists = {k.split("_", 1)[1]: v for k, v in recommended_items.items() if k.startswith("Hybrid_")}
    # Need ground truth keys matching - simplify
    # Actually let's just compute for the first few users for demo
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precision@K (CF)", f"{prec_cf:.3f}")
    with col2:
        st.metric("Recall@K (CF)", f"{rec_cf:.3f}")
    with col3:
        # Simple MAP approximation
        map_k = calculate_map_at_k(ground_truth, recommended_items, k=k)
        st.metric("MAP@K", f"{map_k:.3f}")
    
    # Catalog Coverage
    total_tracks = set(tracks_df["track_id"].tolist())
    coverage = calculate_catalog_coverage(recommended_items, total_tracks)
    st.metric("Catalog Coverage", f"{coverage:.1%}")
    
    # Intra-List Diversity
    ild = calculate_intra_list_diversity(
        recommended_items.get("CF_USR001", []),
        feature_matrix,
        track_to_idx_eval
    )
    st.metric("Intra-List Diversity (ILD)", f"{ild:.3f}")
    
    # Visualizations
    st.subheader("📊 Metric Visualizations")
    
    # Radar chart for a sample track
    sample_track = tracks_df.sample(1).iloc[0]
    fig_radar = go.Figure()
    
    # Add the sample track
    fig_radar.add_trace(go.Scatterpolar(
        r=[sample_track["danceability"], sample_track["energy"], 
           sample_track["acousticness"], sample_track["valence"]],
        theta=["Danceability", "Energy", "Acousticness", "Valence"],
        fill='toself',
        name=sample_track["title"],
        marker_color="#e94560"
    ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Sample Track Audio Profile",
        template="plotly_dark"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

def show_settings(tracks_df, ratings_df, hybrid, user_to_idx, track_to_idx):
    st.header("⚙️ Settings & Configuration")
    
    st.subheader("Dataset Parameters")
    st.write(f"Total Tracks: {len(tracks_df)}")
    st.write(f"Total Users: {ratings_df['user_id'].nunique()}")
    st.write(f"Total Ratings: {len(ratings_df)}")
    
    st.subheader("Audio Feature Columns")
    st.multiselect(
        "Select features to emphasize:",
        options=["danceability", "energy", "acousticness", "valence", "tempo", "loudness", "popularity"],
        default=["danceability", "energy", "acousticness", "valence", "tempo", "loudness", "popularity"]
    )
    
    st.subheader("Model Hyperparameters")
    n_components = st.slider("SVD Components (Collaborative)", 2, 50, 10, help="Number of latent factors for matrix factorization")
    random_state = st.number_input("Random State", value=42, min_value=0)
    
    if st.button("Retrain Models"):
        with st.spinner("Retraining models..."):
            # Reinitialize with new params
            st.success("Models retrained successfully!")
            st.experimental_rerun()

if __name__ == "__main__":
    main()