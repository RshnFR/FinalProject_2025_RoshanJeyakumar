import pandas as pd
import numpy as np
import ast
import os

# --- Config---
DATA_DIR = 'data/'
OUTPUT_DIR = 'data/'
WATCH_RATIO_CAP = 5.0
LIKE_THRESHOLD = 1.5

def load_interaction_data(file_name, data_dir=DATA_DIR, nrows=None, label=""):
    file_path = os.path.join(data_dir, file_name)
    print(f"\nLoading {label} interaction data from: {file_path} (nrows={nrows or 'all'})")
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        print(f"Successfully loaded {file_name}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Make sure it's in the '{data_dir}' directory.")
        return None
    except Exception as e:
        print(f"An error occurred while loading {file_name}: {e}")
        return None

def load_metadata_file(file_name, data_dir=DATA_DIR, converters=None):
    file_path = os.path.join(data_dir, file_name)
    print(f"Loading metadata: {file_path}")
    try:
        return pd.read_csv(file_path, converters=converters)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {file_path}")
        return None

def parse_feat_list_string(x):
    if pd.isnull(x) or not isinstance(x, str) or not x.startswith('['):
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []

def run_common_preprocessing(df, df_label="Interaction Data"):
    print(f"\nPreprocessing {df_label} (Initial shape: {df.shape})")
    if df is None:
        print(f"Skipping preprocessing for {df_label} as it's None.")
        return None

    # 1. Handle Timestamps
    df['datetime_col'] = pd.to_datetime(df['timestamp'], unit='s')
    
    rows_before_drop = len(df)
    df.dropna(subset=['datetime_col'], inplace=True)
    print(f"{df_label}: Dropped {rows_before_drop - len(df)} rows with missing 'datetime_col'. Shape after drop: {df.shape}")

    df['watch_ratio_capped'] = np.clip(df['watch_ratio'], 0, WATCH_RATIO_CAP)

    df['like'] = (df['watch_ratio_capped'] > LIKE_THRESHOLD).astype(int)
    print(f"{df_label} 'like' label distribution:\n{df['like'].value_counts(normalize=True, dropna=False)}")
    
    cols_to_drop = ['time', 'date', 'timestamp', 'watch_ratio']
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            
    print(f"{df_label}: Shape after common preprocessing: {df.shape}")
    return df

if __name__ == "__main__":
    print("--- Script: Preprocessing KuaiRec Data (Train on Big, Test on Small) ---")

    df_train_interactions = load_interaction_data('big_matrix.csv', nrows=None, label="TRAIN (big_matrix)")
    df_test_interactions = load_interaction_data('small_matrix.csv', label="TEST (small_matrix)")

    if df_train_interactions is None or df_test_interactions is None:
        print("\nOne or both interaction files failed to load. Exiting.")
        exit()

    item_categories_df = load_metadata_file('item_categories.csv', converters={'feat': parse_feat_list_string})
    user_features_df = load_metadata_file('user_features.csv')
   
    if item_categories_df is None or user_features_df is None:
        print("\nOne or more metadata files failed to load. Exiting.")
        exit()

    print("\nPreprocessing User Features metadata...")
    onehot_cols = [f'onehot_feat{i}' for i in range(18)]
    existing_onehot_cols = [col for col in onehot_cols if col in user_features_df.columns]
    for col in existing_onehot_cols:
        if user_features_df[col].isnull().any():
            user_features_df[col].fillna(0, inplace=True)
            print(f"Imputed NaNs in user_features_df['{col}'] with 0.")

    if 'feat' in item_categories_df.columns:
        item_categories_df.rename(columns={'feat': 'feat_parsed'}, inplace=True)
        print("Renamed 'feat' to 'feat_parsed' in item_categories_df.")

    df_train_processed_interactions = run_common_preprocessing(df_train_interactions, "TRAIN Interaction Data")
    df_test_processed_interactions = run_common_preprocessing(df_test_interactions, "TEST Interaction Data")

    def merge_with_metadata(df_interactions, item_cats, user_feats, df_label):
        print(f"\nMerging {df_label} with metadata...")
        if df_interactions is None:
            print(f"Skipping merge for {df_label} as interaction data is None.")
            return None
        
        # Merge with item categories
        item_cols_to_merge = ['video_id']
        if 'feat_parsed' in item_cats.columns:
            item_cols_to_merge.append('feat_parsed')
        else:
            print(f"Warning: 'feat_parsed' not found in item_categories for {df_label} merge.")

        merged_df = pd.merge(df_interactions, item_cats[item_cols_to_merge], on='video_id', how='left')
        
        if 'feat_parsed' in merged_df.columns:
            merged_df['feat_parsed'] = merged_df['feat_parsed'].apply(lambda x: x if isinstance(x, list) else [])
        
        # Merge with user features
        merged_df = pd.merge(merged_df, user_feats, on='user_id', how='left')
        print(f"{df_label} shape after all merges: {merged_df.shape}")
        
        print(f"Missing values count in {df_label} after merges (first 5 columns with NaNs):")
        print(merged_df.isnull().sum().sort_values(ascending=False).head())
        
        return merged_df

    train_final_df = merge_with_metadata(df_train_processed_interactions, item_categories_df, user_features_df, "TRAIN Data")
    test_final_df = merge_with_metadata(df_test_processed_interactions, item_categories_df, user_features_df, "TEST Data")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if train_final_df is not None:
        train_output_path = os.path.join(OUTPUT_DIR, 'interactions_train.csv')
        train_final_df.to_csv(train_output_path, index=False)
        print(f"\nProcessed TRAIN data saved to: {train_output_path} (Shape: {train_final_df.shape})")

    if test_final_df is not None:
        test_output_path = os.path.join(OUTPUT_DIR, 'interactions_test.csv')
        test_final_df.to_csv(test_output_path, index=False)
        print(f"Processed TEST data saved to: {test_output_path} (Shape: {test_final_df.shape})")

    sample_submission_df = None
    # 6. Generate Sample Submission Template
    if sample_submission_df is None:
        print("\nGenerating sample submission template...")
        if test_final_df is not None:
            unique_users = test_final_df['user_id'].unique()
            sample_rows = []
            for user_id in unique_users:
                sample_rows.append({
                    'user_id': user_id,
                    'video_id': [],
                    'prediction_score': []
                })
            
            sample_submission_df = pd.DataFrame(sample_rows)
            sample_submission_path = os.path.join(OUTPUT_DIR, 'sample_submission.csv')
            sample_submission_df.to_csv(sample_submission_path, index=False)
            print(f"Generated sample submission template saved to: {sample_submission_path}")
        else:
            print("Cannot generate sample submission without test data.")
    else:
        sample_submission_path = os.path.join(OUTPUT_DIR, 'sample_submission_reference.csv')
        sample_submission_df.to_csv(sample_submission_path, index=False)
        print(f"Sample submission reference saved to: {sample_submission_path}")
    print("\n--- Preprocessing script finished. ---")
