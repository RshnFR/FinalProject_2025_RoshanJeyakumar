import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score
from joblib import load
import os

# --- Config ---
DATA_DIR = 'data/'
PROCESSED_TEST_FILE = os.path.join(DATA_DIR, 'test_processed.csv')
MODEL_PATH = os.path.join(DATA_DIR, 'lightfm_improved_model.pkl')
FEATURE_COLUMNS_PATH = os.path.join(DATA_DIR, 'feature_columns_big_train_small_test.txt')
PREDICTIONS_FILE_PATH = os.path.join(DATA_DIR, 'lightfm_improved_predictions_test_set.csv')

K_VALUES = [5, 10, 20]

def precision_at_k(y_true_relevant, y_pred_top_k):
    """Calculates Precision@K.
    y_true_relevant: set of true relevant items.
    y_pred_top_k: list of top K recommended items.
    """
    relevant_and_recommended = len(set(y_pred_top_k) & y_true_relevant)
    return relevant_and_recommended / len(y_pred_top_k) if len(y_pred_top_k) > 0 else 0.0

def recall_at_k(y_true_relevant, y_pred_top_k):
    """Calculates Recall@K.
    y_true_relevant: set of true relevant items.
    y_pred_top_k: list of top K recommended items.
    """
    relevant_and_recommended = len(set(y_pred_top_k) & y_true_relevant)
    return relevant_and_recommended / len(y_true_relevant) if len(y_true_relevant) > 0 else 0.0

def average_precision_at_k(y_true_relevant, y_pred_ranked_k):
    """Calculates Average Precision@K (AP@K).
    y_true_relevant: set of true relevant items.
    y_pred_ranked_k: list of K recommended items, in ranked order.
    """
    if not y_true_relevant:
        return 0.0
    
    ap = 0.0
    hits = 0
    for i, p in enumerate(y_pred_ranked_k):
        if p in y_true_relevant:
            hits += 1
            ap += hits / (i + 1.0)
            
    return ap / len(y_true_relevant) if y_true_relevant else 0.0


def ndcg_at_k(y_true_relevance_scores_map, y_pred_ranked_k):
    """Calculates Normalized Discounted Cumulative Gain (NDCG@K).
    y_true_relevance_scores_map: dict of {item_id: relevance_score} for true relevant items.
                                      For binary relevance, score is 1 if relevant, 0 otherwise.
    y_pred_ranked_k: list of K recommended items, in ranked order.
    """
    k = len(y_pred_ranked_k)
    dcg = 0.0
    for i, item_id in enumerate(y_pred_ranked_k):
        relevance = y_true_relevance_scores_map.get(item_id, 0)
        dcg += relevance / np.log2(i + 2)


    ideal_ranked_true_items = sorted(
        [item_id for item_id in y_true_relevance_scores_map if y_true_relevance_scores_map[item_id] > 0],
        key=lambda x: y_true_relevance_scores_map[x],
        reverse=True
    )[:k]

    idcg = 0.0
    for i, item_id in enumerate(ideal_ranked_true_items):
        relevance = y_true_relevance_scores_map.get(item_id, 0)
        idcg += relevance / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


if __name__ == "__main__":
    print("--- Script: Recommender System Evaluation ---")

    all_user_predictions_df = None
    try:
        print(f"Attempting to load predictions from: {PREDICTIONS_FILE_PATH}")
        all_user_predictions_df = pd.read_csv(PREDICTIONS_FILE_PATH)
        print(f"Successfully loaded predictions. Shape: {all_user_predictions_df.shape}")
    except FileNotFoundError:
        print(f"Predictions file '{PREDICTIONS_FILE_PATH}' not found. Will attempt to regenerate.")
    except Exception as e:
        print(f"Error loading predictions file: {e}. Will attempt to regenerate.")

    if all_user_predictions_df is None:
        print("\nRegenerating predictions as they were not loaded...")
        print(f"Loading processed test data from: {PROCESSED_TEST_FILE}")
        df_test_raw = pd.read_csv(PROCESSED_TEST_FILE, usecols=['user_id', 'video_id', 'like'])

        print(f"Loading trained model from: {MODEL_PATH}")
        model = load(MODEL_PATH)
        print(f"Loading feature columns from: {FEATURE_COLUMNS_PATH}")
        with open(FEATURE_COLUMNS_PATH, 'r') as f:
            feature_columns = [line.strip() for line in f if line.strip()]
        

        print("Loading full processed test data for feature engineering...")
        df_test_full_for_fe = pd.read_csv(PROCESSED_TEST_FILE)


        print("Critical: For full regeneration, feature engineering state (learned OHE columns, top tags) is needed.")
        print("This script primarily expects 'predictions_big_train_small_test.csv' to exist.")
        print("If you need to regenerate predictions fully, ensure the FE logic is callable and state is loaded.")

        if not os.path.exists(PREDICTIONS_FILE_PATH):
            print(f"ERROR: {PREDICTIONS_FILE_PATH} not found. Please ensure the previous script "
                  f"feature_engineer_and_train.py ran successfully and created this file (it was an optional save).")
            print("You might need to uncomment the save line in that script for 'ranked_test_recommendations'.")
            print("Example save line in previous script: "
                  "ranked_test_recommendations.to_csv('data/predictions_big_train_small_test.csv', index=False)")
            exit()
        else:
             all_user_predictions_df = pd.read_csv(PREDICTIONS_FILE_PATH)


    if 'actual_like' not in all_user_predictions_df.columns:
        print("Merging 'actual_like' from processed test data.")
        df_test_likes = pd.read_csv(PROCESSED_TEST_FILE, usecols=['user_id', 'video_id', 'like'])
        df_test_likes.rename(columns={'like': 'actual_like'}, inplace=True)
        all_user_predictions_df = pd.merge(all_user_predictions_df, df_test_likes, on=['user_id', 'video_id'], how='left')

    if all_user_predictions_df['actual_like'].isnull().any():
        print("Warning: Some 'actual_like' values are missing after merge. Check user/video IDs.")
        all_user_predictions_df.dropna(subset=['actual_like'], inplace=True)

    all_user_predictions_df.sort_values(
        by=['user_id', 'predicted_like_probability'],
        ascending=[True, False],
        inplace=True
    )

    user_metrics = []
    unique_users = all_user_predictions_df['user_id'].unique()
    print(f"\nCalculating ranking metrics for {len(unique_users)} users...")

    for user_id in unique_users:
        user_data = all_user_predictions_df[all_user_predictions_df['user_id'] == user_id]
        
        # Ground truth: set of video_ids the user actually liked
        y_true_relevant_items = set(user_data[user_data['actual_like'] == 1]['video_id'])
        
        # For NDCG, map of {item_id: relevance_score (1 for like, 0 for dislike)}
        # Considering only interacted items for relevance map
        y_true_relevance_map = pd.Series(user_data['actual_like'].values, index=user_data['video_id']).to_dict()

        # Recommended items (ranked by probability)
        y_pred_ranked_all = user_data['video_id'].tolist()

        user_k_metrics = {'user_id': user_id}
        for k in K_VALUES:
            y_pred_top_k_items = y_pred_ranked_all[:k]
            
            user_k_metrics[f'P@{k}'] = precision_at_k(y_true_relevant_items, y_pred_top_k_items)
            user_k_metrics[f'R@{k}'] = recall_at_k(y_true_relevant_items, y_pred_top_k_items)
            user_k_metrics[f'AP@{k}'] = average_precision_at_k(y_true_relevant_items, y_pred_top_k_items) # AP uses all ranked items up to k
            user_k_metrics[f'NDCG@{k}'] = ndcg_at_k(y_true_relevance_map, y_pred_top_k_items)
        user_metrics.append(user_k_metrics)

    # Convert list of user metrics to DataFrame
    user_metrics_df = pd.DataFrame(user_metrics)

    # Calculate average metrics across all users
    mean_metrics = {}
    for k in K_VALUES:
        mean_metrics[f'Mean P@{k}'] = user_metrics_df[f'P@{k}'].mean()
        mean_metrics[f'Mean R@{k}'] = user_metrics_df[f'R@{k}'].mean()
        mean_metrics[f'MAP@{k}'] = user_metrics_df[f'AP@{k}'].mean() # MAP is mean of APs
        mean_metrics[f'Mean NDCG@{k}'] = user_metrics_df[f'NDCG@{k}'].mean()
        
    print("\n--- Overall Recommender Evaluation Metrics ---")
    for metric_name, value in mean_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Print a sample of user-specific metrics
    print("\nSample of user-specific metrics (first 5 users):")
    print(user_metrics_df.head())

    print("\n--- Recommender Evaluation script finished. ---")
