import pandas as pd
import numpy as np
import ast
import os
from collections import Counter
from lightfm import LightFM
from lightfm.data import Dataset
import pickle

# --- Configuration ---
DATA_DIR = 'data/'
PROCESSED_TRAIN_FILE = os.path.join(DATA_DIR, 'interactions_train.csv')
PROCESSED_TEST_FILE = os.path.join(DATA_DIR, 'interactions_test.csv')
MODEL_OUTPUT_PATH = os.path.join(DATA_DIR, 'lightfm_improved_model.pkl')
PREDICTIONS_OUTPUT_CSV_PATH = os.path.join(DATA_DIR, 'lightfm_improved_predictions_test_set.csv')

TOP_N_TAGS_TO_USE = 30 
TRAIN_SAMPLE_RATIO = 1 

# LightFM Hyperparameters
LFM_NO_COMPONENTS = 8
LFM_LEARNING_RATE = 0.05
LFM_LOSS = 'warp'
LFM_ITEM_ALPHA = 1e-6 # Regularization for item features
LFM_USER_ALPHA = 1e-6 # Regularization for user features
LFM_EPOCHS = 10      
LFM_MAX_SAMPLED = 15  

learned_top_tags_list_lfm = []
learned_ohe_columns_map_lfm = {}

def parse_list_string_if_needed(val):
    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
        try: return ast.literal_eval(val)
        except: return []
    elif isinstance(val, list): return val
    return []

# --- Feature Engineering ---
def engineer_base_features(df, df_label, is_train_set=False):
    global learned_top_tags_list_lfm, learned_ohe_columns_map_lfm
    
    print(f"\nEngineering base features for {df_label} data (Shape: {df.shape})")
    features_df = df.copy()

    if 'datetime_col' in features_df.columns:
        features_df['datetime_col'] = pd.to_datetime(features_df['datetime_col'], errors='coerce')
        features_df['hour_of_day'] = features_df['datetime_col'].dt.hour
        features_df['day_of_week'] = features_df['datetime_col'].dt.dayofweek
        features_df.drop('datetime_col', axis=1, inplace=True)
    else: print(f"{df_label}: No 'datetime_col'.")

    categorical_cols_to_encode = ['user_active_degree']
    for col in features_df.select_dtypes(include='object').columns:
        if '_range' in col and col not in categorical_cols_to_encode:
            categorical_cols_to_encode.append(col)
    categorical_cols_to_encode = [col for col in categorical_cols_to_encode if col in features_df.columns]
    
    for col_name in categorical_cols_to_encode:
        features_df[col_name] = features_df[col_name].astype(str).fillna('Missing')
        if is_train_set:
            dummies = pd.get_dummies(features_df[col_name], prefix=col_name, dummy_na=False).astype(int)
            learned_ohe_columns_map_lfm[col_name] = dummies.columns.tolist()
            features_df = pd.concat([features_df, dummies], axis=1)
        else:
            if col_name in learned_ohe_columns_map_lfm:
                current_dummies = pd.get_dummies(features_df[col_name], prefix=col_name, dummy_na=False).astype(int)
                aligned_dummies = pd.DataFrame(columns=learned_ohe_columns_map_lfm[col_name], index=features_df.index).fillna(0)
                for c_col in current_dummies.columns:
                    if c_col in aligned_dummies.columns:
                        aligned_dummies[c_col] = current_dummies[c_col].reindex(aligned_dummies.index).fillna(0)
                features_df = pd.concat([features_df, aligned_dummies], axis=1)
            else: print(f"{df_label}: Warn! OHE for '{col_name}' not learned.")
        features_df.drop(col_name, axis=1, inplace=True)

    if 'feat_parsed' in features_df.columns:
        features_df['feat_parsed'] = features_df['feat_parsed'].apply(parse_list_string_if_needed)
        def sanitize_tag_list(tl): return [int(t) for t in tl if str(t).isdigit()] if isinstance(tl, list) else []
        features_df['feat_parsed'] = features_df['feat_parsed'].apply(sanitize_tag_list)
        if is_train_set:
            tags = [t for sl in features_df['feat_parsed'] for t in sl]
            learned_top_tags_list_lfm.clear()
            learned_top_tags_list_lfm.extend([t for t,c in Counter(tags).most_common(TOP_N_TAGS_TO_USE)])
            print(f"{df_label}: Learned top {len(learned_top_tags_list_lfm)} tags.")
        if learned_top_tags_list_lfm:
            for tid in learned_top_tags_list_lfm: features_df[f'tag_{tid}'] = features_df['feat_parsed'].apply(lambda x: 1 if tid in x else 0)
        features_df.drop('feat_parsed', axis=1, inplace=True)
    else: print(f"{df_label}: No 'feat_parsed'.")
        
    print(f"{df_label}: Shape after base FE: {features_df.shape}")
    return features_df

if __name__ == "__main__":
    print("--- Script: Improved LightFM Feature Engineering and Model Training ---")

    # 1. Load Data
    print(f"Loading train data: {PROCESSED_TRAIN_FILE}")
    df_train_full = pd.read_csv(PROCESSED_TRAIN_FILE, low_memory=False)
    if TRAIN_SAMPLE_RATIO < 1.0 and 0 < TRAIN_SAMPLE_RATIO:
        print(f"Sampling {TRAIN_SAMPLE_RATIO*100:.1f}% of training data.")
        df_train = df_train_full.sample(frac=TRAIN_SAMPLE_RATIO, random_state=42).copy()
    else:
        df_train = df_train_full.copy()
    del df_train_full

    print(f"Loading test data: {PROCESSED_TEST_FILE}")
    df_test = pd.read_csv(PROCESSED_TEST_FILE, low_memory=False)
    print(f"Initial train data shape: {df_train.shape}, Test data shape: {df_test.shape}")

    # 2. Base Features
    train_base_features_df = engineer_base_features(df_train, "TRAIN", is_train_set=True)
    test_base_features_df = engineer_base_features(df_test, "TEST", is_train_set=False)
    del df_train, df_test

    # 3. Item Feature
    lightfm_item_feature_columns = [col for col in train_base_features_df.columns if col.startswith('tag_')]
    if 'video_duration' in train_base_features_df.columns:
         lightfm_item_feature_columns.append('video_duration')
   
    for col in lightfm_item_feature_columns:
        if col not in test_base_features_df.columns:
             print(f"Item feature '{col}' missing in TEST base features, adding with 0.")
             test_base_features_df[col] = 0


    print(f"\nSelected {len(lightfm_item_feature_columns)} item feature columns for LightFM: {lightfm_item_feature_columns[:5]}...")

    # 4. Prepare Data
    print("\n--- Preparing Data for LightFM ---")
    all_user_ids_lfm = pd.concat([train_base_features_df['user_id'], test_base_features_df['user_id']]).unique()
    all_item_ids_lfm = pd.concat([train_base_features_df['video_id'], test_base_features_df['video_id']]).unique()
    
    dataset_lfm = Dataset()
    dataset_lfm.fit(
        users=all_user_ids_lfm, 
        items=all_item_ids_lfm, 
        item_features=lightfm_item_feature_columns
    )
    
    df_train_interactions_source = pd.read_csv(PROCESSED_TRAIN_FILE, usecols=['user_id', 'video_id', 'watch_ratio_capped'], low_memory=False)
    if TRAIN_SAMPLE_RATIO < 1.0 and 0 < TRAIN_SAMPLE_RATIO:
        df_train_interactions_source = df_train_interactions_source.sample(frac=TRAIN_SAMPLE_RATIO, random_state=42).copy()


    df_train_interactions_lfm = df_train_interactions_source[df_train_interactions_source['watch_ratio_capped'] > 0.01].copy()
    
    print(f"Building train interactions for LightFM using 'watch_ratio_capped' (found {len(df_train_interactions_lfm)} valid interactions)...")
    (train_interactions_lfm, train_weights_lfm) = dataset_lfm.build_interactions(
        (row['user_id'], row['video_id'], float(row['watch_ratio_capped']))
        for index, row in df_train_interactions_lfm.iterrows()
    )
    del df_train_interactions_source, df_train_interactions_lfm

    print("Building item features for LightFM...")

    unique_items_for_features_df = train_base_features_df.drop_duplicates(subset=['video_id'], keep='first')
    
    item_features_lfm_matrix = dataset_lfm.build_item_features(
        (row['video_id'], {feat_col: row[feat_col] for feat_col in lightfm_item_feature_columns if feat_col in row and pd.notna(row[feat_col])})
        for index, row in unique_items_for_features_df.iterrows()
    )
    del unique_items_for_features_df

    print(f"LightFM Train interactions shape: {train_interactions_lfm.shape}")
    print(f"LightFM Item features shape: {item_features_lfm_matrix.shape}")
    
    user_features_lfm_matrix = None 

    # 5. Train LightFM Model
    print("\n--- Training Improved LightFM Model ---")
    model_lfm = LightFM(
        loss=LFM_LOSS,
        learning_rate=LFM_LEARNING_RATE,
        no_components=LFM_NO_COMPONENTS,
        item_alpha=LFM_ITEM_ALPHA,
        user_alpha=LFM_USER_ALPHA,
        max_sampled=LFM_MAX_SAMPLED,
        random_state=42
    )
    print("Fitting LightFM model...")
    model_lfm.fit(
        train_interactions_lfm,
        sample_weight=train_weights_lfm,
        item_features=item_features_lfm_matrix,
        user_features=user_features_lfm_matrix,
        epochs=LFM_EPOCHS,
        num_threads=1, 
        verbose=True
    )
    print("LightFM Model training complete.")
    del train_interactions_lfm, train_weights_lfm, item_features_lfm_matrix

    # 6. Prepare for Prediction on Test Set
    print("\n--- Generating predictions for TEST set using Improved LightFM ---")
    user_id_map_lfm, _, item_id_map_lfm, _ = dataset_lfm.mapping()
    
    predictions_lfm_df = test_base_features_df[['user_id', 'video_id', 'like']].copy()
    predictions_lfm_df.rename(columns={'like': 'actual_like'}, inplace=True)
    
    all_known_items_features_df = pd.concat([
        train_base_features_df[['video_id'] + lightfm_item_feature_columns],
        test_base_features_df[['video_id'] + lightfm_item_feature_columns]
    ]).drop_duplicates(subset=['video_id'], keep='first')

    item_features_for_prediction = dataset_lfm.build_item_features(
        (row['video_id'], {feat_col: row[feat_col] for feat_col in lightfm_item_feature_columns if feat_col in row and pd.notna(row[feat_col])})
        for index, row in all_known_items_features_df.iterrows()
    )
    del train_base_features_df, test_base_features_df, all_known_items_features_df

    print("Predicting scores for test set pairs...")
    scores = []
    for index, row in predictions_lfm_df.iterrows():
        raw_user_id = row['user_id']
        raw_item_id = row['video_id']
        
        if raw_user_id in user_id_map_lfm and raw_item_id in item_id_map_lfm:
            internal_user_id = user_id_map_lfm[raw_user_id]
            internal_item_id = item_id_map_lfm[raw_item_id]
            
            score = model_lfm.predict(
                internal_user_id,
                np.array([internal_item_id]),
                item_features=item_features_for_prediction, 
                user_features=user_features_lfm_matrix,
                num_threads=1
            )[0]
            scores.append(score)
        else:
            scores.append(np.nan)

    predictions_lfm_df['predicted_like_probability'] = scores
    predictions_lfm_df.dropna(subset=['predicted_like_probability'], inplace=True)
    predictions_lfm_df['predicted_like_label'] = (predictions_lfm_df['predicted_like_probability'] > 0).astype(int)
    
    ranked_lfm_predictions = predictions_lfm_df.sort_values(
        by=['user_id', 'predicted_like_probability'], ascending=[True, False]
    )
    print("\nSample of Top-5 ranked Improved LightFM predictions:")
    print(ranked_lfm_predictions.groupby('user_id').head(5).head(10))

    # 7. Save
    print(f"\nSaving Improved LightFM model and dataset to: {MODEL_OUTPUT_PATH}")
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump({'model': model_lfm, 'dataset': dataset_lfm}, f)
    
    ranked_lfm_predictions.to_csv(PREDICTIONS_OUTPUT_CSV_PATH, index=False)
    print(f"\nImproved LightFM Ranked test predictions saved to: {PREDICTIONS_OUTPUT_CSV_PATH}")
    
    print("\n--- Improved LightFM Script Finished ---")

