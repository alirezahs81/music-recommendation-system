"""
Evaluation Metrics Module for Recommendation Systems.
Calculates RMSE, MAE, Precision@K, Recall@K, MAP@K, Catalog Coverage, Novelty, and Intra-List Diversity.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

def calculate_rmse_mae(actual_ratings: np.ndarray, predicted_ratings: np.ndarray) -> Tuple[float, float]:
    """
    Calculates Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
    """
    rmse = float(np.sqrt(mean_squared_error(actual_ratings, predicted_ratings)))
    mae = float(mean_absolute_error(actual_ratings, predicted_ratings))
    return rmse, mae

def calculate_precision_recall_at_k(
    relevant_items: Dict[str, Set[str]],
    recommended_items: Dict[str, List[str]],
    k: int = 10
) -> Tuple[float, float]:
    """
    Computes average Precision@K and Recall@K across all evaluated users.
    relevant_items: dict mapping user_id -> set of ground truth liked item IDs
    recommended_items: dict mapping user_id -> list of top-K recommended item IDs
    """
    precisions = []
    recalls = []

    for user_id, recs in recommended_items.items():
        top_k_recs = recs[:k]
        relevant = relevant_items.get(user_id, set())

        if not relevant:
            continue

        hits = len(set(top_k_recs).intersection(relevant))
        precision = hits / len(top_k_recs) if top_k_recs else 0.0
        recall = hits / len(relevant) if relevant else 0.0

        precisions.append(precision)
        recalls.append(recall)

    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    return mean_precision, mean_recall

def calculate_map_at_k(
    relevant_items: Dict[str, Set[str]],
    recommended_items: Dict[str, List[str]],
    k: int = 10
) -> float:
    """
    Calculates Mean Average Precision at K (MAP@K).
    """
    average_precisions = []

    for user_id, recs in recommended_items.items():
        top_k_recs = recs[:k]
        relevant = relevant_items.get(user_id, set())

        if not relevant:
            continue

        hits = 0
        sum_precisions = 0.0
        for i, item in enumerate(top_k_recs):
            if item in relevant:
                hits += 1
                sum_precisions += hits / (i + 1)

        ap = sum_precisions / min(len(relevant), k) if relevant else 0.0
        average_precisions.append(ap)

    return float(np.mean(average_precisions)) if average_precisions else 0.0

def calculate_catalog_coverage(
    recommended_items: Dict[str, List[str]],
    total_items: Set[str]
) -> float:
    """
    Calculates the proportion of items in the total catalog recommended at least once.
    """
    all_recommended = set()
    for recs in recommended_items.values():
        all_recommended.update(recs)

    if not total_items:
        return 0.0

    return float(len(all_recommended) / len(total_items))

def calculate_intra_list_diversity(
    recommended_track_ids: List[str],
    feature_matrix: np.ndarray,
    track_to_idx: Dict[str, int]
) -> float:
    """
    Computes Intra-List Diversity (ILD) as 1 - mean cosine similarity among recommended items.
    """
    if len(recommended_track_ids) < 2:
        return 0.0

    indices = [track_to_idx[t] for t in recommended_track_ids if t in track_to_idx]
    if len(indices) < 2:
        return 0.0

    item_vectors = feature_matrix[indices]
    sim_matrix = cosine_similarity(item_vectors)

    # Compute mean similarity among off-diagonal elements
    n = len(indices)
    total_sim = (np.sum(sim_matrix) - n) / (n * (n - 1))
    return float(1.0 - total_sim)
