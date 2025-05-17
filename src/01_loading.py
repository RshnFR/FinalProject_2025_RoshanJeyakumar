import pandas as pd
import os

# --- Config---
DATA_DIR = './data/'

def load_and_inspect(file_name, data_dir=DATA_DIR,nrows=None):
    """Loads a CSV file and performs basic inspection."""
    file_path = os.path.join(data_dir, file_name)
    print(f"\n--- Loading and Inspecting: {file_name} ---")
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        print(f"Successfully loaded {file_name}")
        
        print("\nShape:")
        print(df.shape)
        
        print("\nHead:")
        print(df.head())
        
        print("\nInfo:")
        df.info()
        
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        print("\nBasic Statistics (Numerical):")
        print(df.describe())
        
        print("\nBasic Statistics (All dtypes):")
        print(df.describe(include='all'))
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while processing {file_name}: {e}")
        return None

# --- 1.Loading and Inspection ---

print("Attempting to load small_matrix.csv...")
interactions_df = load_and_inspect('small_matrix.csv')

if interactions_df is not None:
    print("\n--- Specific Inspections for interactions_df (small_matrix.csv) ---")
    print("\nOriginal 'time' and 'timestamp' data types:")
    print(interactions_df[['time', 'timestamp']].head())
    
    interactions_df['datetime_from_time'] = pd.to_datetime(interactions_df['time'])
    interactions_df['datetime_from_timestamp'] = pd.to_datetime(interactions_df['timestamp'], unit='s')
    interactions_df['date_yyyymmdd'] = pd.to_datetime(interactions_df['date'], format='%Y%m%d')

    print("\nConverted datetime columns head:")
    print(interactions_df[['datetime_from_time', 'datetime_from_timestamp', 'date_yyyymmdd']].head())
    
    print("\nInfo after datetime conversion:")
    interactions_df.info()
    
    print("\nDistribution of 'watch_ratio':")
    print(interactions_df['watch_ratio'].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))
    
    print("\nDistribution of 'play_duration' (ms):")
    print(interactions_df['play_duration'].describe())
    
    print("\nDistribution of 'video_duration' (ms):")
    print(interactions_df['video_duration'].describe())
    
    calculated_watch_ratio = interactions_df['play_duration'] / interactions_df['video_duration']
    comparison = pd.DataFrame({
        'original_watch_ratio': interactions_df['watch_ratio'],
        'calculated_watch_ratio': calculated_watch_ratio,
        'difference': interactions_df['watch_ratio'] - calculated_watch_ratio
    })
    print("\nComparison of original vs. calculated 'watch_ratio':")
    print(comparison.describe())
    print("Number of rows with significant difference (e.g., > 0.01):")
    print((comparison['difference'].abs() > 0.01).sum())


# Load item_categories.csv
print("\nAttempting to load item_categories.csv...")
item_categories_df = load_and_inspect('item_categories.csv')
if item_categories_df is not None:
    print("\n--- Specific Inspections for item_categories_df ---")
    print("Example of 'feat' column (will need parsing):")
    print(item_categories_df['feat'].head())


# Load user_features.csv
print("\nAttempting to load user_features.csv...")
user_features_df = load_and_inspect('user_features.csv')
if user_features_df is not None:
    print("\n--- Specific Inspections for user_features_df ---")
    onehot_cols = [f'onehot_feat{i}' for i in range(18)]
    print(f"\nUnique value counts for an example one-hot feature ('onehot_feat0'):")
    if 'onehot_feat0' in user_features_df.columns:
        print(user_features_df['onehot_feat0'].value_counts())
    else:
        print("'onehot_feat0' not found.")
        
print("\n--- Initial Data Loading and Inspection Phase Complete ---")
print("Next steps would involve deeper cleaning, merging, and feature engineering.")

