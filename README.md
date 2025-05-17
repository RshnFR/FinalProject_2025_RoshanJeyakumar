# KuaiRec Short Video Recommender System

## Project Overview

This project aims to develop a personalized and scalable short video recommender system using the KuaiRec dataset, collected from the Kuaishou platform. The dataset features fully observed user-item interactions and rich metadata. The primary goal is to build a recommendation engine that accurately predicts which videos users are likely to enjoy and engage with deeply.

The project involves data loading, extensive preprocessing, feature engineering, model development using the LightFM framework, and rigorous evaluation of recommendation quality.

## Dataset Description

The KuaiRec dataset provides a comprehensive view of user interactions and associated metadata:

*   **Interaction Matrices (`big_matrix.csv`, `small_matrix.csv`)**: Core user-video interaction logs.
    *   Fields: `user_id`, `video_id`, `play_duration`, `video_duration`, `time`, `date`, `timestamp`.
    *   Key derived metric: `watch_ratio` (`play_duration` / `video_duration`), serving as an implicit feedback signal.
*   **Video Features**:
    *   `item_categories.csv`: Video tags/categories (`feat`).
    *   `item_daily_features.csv`: Daily video statistics including `author_id`, `upload_dt`, engagement counts (`show_cnt`, `play_cnt`, `like_cnt`), etc.
    *   `kuairec_caption_category.csv`: Textual information like `manual_cover_text`, `caption`, `topic_tag`, and multi-level category names.
*   **User Features (`user_features.csv`)**: User metadata.
    *   Fields: `user_active_degree`, `is_live_streamer`, social graph sizes (`follow_user_num`, `fans_user_num`), `register_days`, and encrypted one-hot encoded categorical features (`onehot_feat0` to `onehot_feat17`).

## Project Pipeline & Methodology

The project followed a structured pipeline:

### 1. Data Loading and Initial Inspection (Script: `01_loading.py`)

*   All CSV files were loaded into pandas DataFrames.
*   Initial inspection involved checking data types, identifying missing values, and computing basic statistics for each dataset.
*   Timestamps were converted to datetime objects for easier manipulation and analysis.
*   The distribution of `watch_ratio` was analyzed to understand user engagement patterns. Anomalously high `watch_ratio` values (e.g., > 500) were noted, suggesting potential outliers or very short videos with repeated plays.

### 2. Data Preprocessing (Script: `02_preprocess.py`)

*   **Train-Test Split**: `big_matrix.csv` was designated as the primary training set, and `small_matrix.csv` as the test set for interactions.
*   **`watch_ratio` Capping**: To handle extreme outliers observed in `watch_ratio`, values were capped at 10.0 (denoted as `watch_ratio_capped`). This helps stabilize model training.
*   **Feature Cleaning and Transformation**:
    *   **User Features**: Focused on a subset of features based on iterative experiments. For instance, one experiment used only `onehot_feat0` to `onehot_feat17`, ensuring NaNs were filled (e.g., with 0).
    *   **Item Daily Features**: Aggregated key metrics like `play_cnt` (sum and mean over time for each video) to create static item popularity features.
    *   **Item Categories/Tags**: Parsed string representations of tag lists (e.g., `[tag1, tag2]`) into actual Python lists for easier feature extraction.
    *   **Caption & Category Data**: Selected textual features (captions, topic tags, category names). `video_id` in this file required careful type conversion due to initial parsing as objects and some non-numeric entries. `topic_tag` was also processed into a list format.
*   **Merging**: Interaction data was enriched by merging with the processed user and item metadata using `user_id` and `video_id` as keys.
*   **NaN Handling Post-Merge**:
    *   Numerical features (e.g., aggregated play counts, one-hot features) were filled with 0.
    *   List-based features (e.g., `feat_list`, `topic_tag_list`) had NaNs replaced with empty lists.
    *   Categorical text features (e.g., category names, captions) had NaNs replaced with "Unknown".
*   **Output**: Processed DataFrames (`train_processed_*.pkl`, `test_processed_*.pkl`) were saved to disk to avoid re-running preprocessing.

### 3. Feature Engineering & Model Development (Script: `03_model.py`)

*   **Rationale for LightFM**: LightFM was selected for its ability to:
    *   Implement a **hybrid recommendation model**, combining collaborative filtering (from interaction data) with content-based filtering (from user/item features). This is ideal for the rich KuaiRec dataset.
    *   Effectively handle **implicit feedback** (like `watch_ratio`) using appropriate loss functions (e.g., WARP).
    *   Incorporate diverse side information (user and item metadata) to improve personalization and address cold-start scenarios.
    *   Scale to large datasets.
*   **LightFM Dataset Construction**:
    *   A `Dataset` object was created and fitted with unique user IDs and item IDs from the training set.
    *   User and item features were then fitted to the dataset. Features were transformed into strings of the format `feature_name:feature_value` (e.g., `play_cnt_sum_bin:3`, `onehot_feat0:1.0`, `first_level_category_name:Music`).
    *   Numerical features like aggregated play counts were binned (e.g., using `pd.qcut`) before being converted to categorical-like features for LightFM.
    *   High-cardinality text features (e.g., raw `caption`) were noted as a challenge; for simplicity, they were initially treated as categorical features, which can lead to a very large feature space. Experiments involved varying the set of included features.
*   **Interaction and Feature Matrices**:
    *   The main interaction matrix (users vs. items) was built using `user_id`, `video_id`, and `watch_ratio_capped` as weights.
    *   Sparse user-feature and item-feature matrices were constructed.
*   **Model Training (Iterative Process)**:
    *   The LightFM model was initialized with parameters such as `no_components` (embedding dimensionality, experimented with values like 10), `learning_rate` (e.g., 0.05), `loss` function (typically 'warp'), and regularization terms (`item_alpha`, `user_alpha`, e.g., 1e-6).
    *   Training was performed using `model.fit_partial()` in a loop for a specified number of epochs. This allowed for **epoch-wise evaluation** on the test set to monitor performance (Precision@K, Recall@K, NDCG@K) and detect overfitting.
    *   Experiments included:
        1.  Training with a comprehensive set of user and item features.
        2.  Simplifying to train *only* with item play count features as a baseline.
*   **Saving Artifacts**: The trained model, LightFM dataset object (containing ID mappings), generated feature matrices, and epoch-wise metrics were saved to disk for later evaluation and analysis.

### 4. Recommendation Generation & Evaluation (Script: `04_evaluate.py` and within `03_model.py`)

*   **Metrics Selection**:
    *   **Precision@K (P@K)**: Proportion of recommended items in the top-K set that are relevant. Measures the exactness of the top recommendations.
    *   **Recall@K (R@K)**: Proportion of all relevant items found in the top-K recommendations. Measures the completeness of recommendations.
    *   **Mean Average Precision@K (MAP@K)**: Average Precision across users, sensitive to the rank of relevant items.
    *   **Normalized Discounted Cumulative Gain@K (NDCG@K)**: Evaluates ranking quality by giving higher scores to relevant items appearing earlier, considering graded relevance (though typically used with binary relevance here based on a "like" threshold).
    These metrics provide a holistic view of recommendation performance.
*   **Definition of Relevance ("Like")**: For calculating P@K, R@K, and MAP@K, a binary definition of "like" was established. An important experiment involved setting a high threshold, e.g., `watch_ratio_capped >= 1.8`, to focus on very strong engagement signals. The `watch_ratio_capped` (or its binary version) was used for NDCG relevance.
*   **Evaluation Scenarios**:
    1.  **Model-Based Evaluation**: The primary goal was to evaluate the LightFM model's predictions. This involved generating top-N recommendations for each test user and comparing against their ground truth interactions in the test set.
    2.  **Oracle Ranking Evaluation (by `watch_ratio_capped`)**: To understand the potential upper bound and the characteristics of the data, an evaluation was performed where items were ranked directly by their actual `watch_ratio_capped` in the test set. This "oracle" ranking was then evaluated against the binary "like" threshold.
    3.  **Random Baseline**: A random recommendation strategy was implemented and evaluated to provide a baseline against which to compare the model and oracle rankings, demonstrating the actual lift achieved.
*   **Epoch-wise Monitoring**: During model training (`03_model.py`), metrics were calculated on the test set after each epoch. This was crucial for observing learning progress and identifying overfitting (e.g., when test set performance starts to degrade while training loss might still be improving).

## Experimental Results & Insights

*   **Initial Model Performance (Complex Features)**: Early iterations with a comprehensive feature set showed significant overfitting after the first epoch. For example, P@10 dropped from ~0.21 to ~0.08 between epoch 1 and 2, with NDCG@10 also falling. This highlighted the need for stronger regularization, careful feature selection (especially for high-cardinality text features like 'caption'), or more sophisticated feature engineering.
*   **Simplified Model (Play Count Only)**: An experiment training only with binned item play counts (and no explicit user features) surprisingly resulted in consistent 0.0 metrics across all epochs. Debugging revealed that while the model was generating varied scores and recommendations, there was a complete misalignment between these popularity-based recommendations and the actual items users interacted with in the test set. This indicated that global historical popularity (as captured by play counts from the training period) was not a sufficient predictor for the specific interactions in the test set.
*   **Oracle Ranking (by `watch_ratio_capped >= 1.8` for "like")**:
    *   This strategy yielded strong results:
        *   Mean P@5: 0.5459, Mean P@10: 0.4962
        *   Mean NDCG@5: 0.5764, Mean NDCG@10: 0.5289
    *   Recall and MAP were lower, which is expected given the very high "like" threshold (1.8) – it's hard to recall all such super-liked items in the top K.
    *   Significantly outperformed a random baseline (e.g., ~4-5x improvement in P@10 and NDCG@10).
    *   This demonstrates that `watch_ratio_capped` is a powerful signal for identifying highly engaging content. The challenge for a predictive model is to learn to rank items similarly *before* the user's true `watch_ratio` is observed.

## Challenges and Future Work

*   **Overfitting**: A primary challenge when using rich feature sets. Requires careful regularization, feature selection/engineering, and potentially more data or different model architectures.
*   **Feature Engineering for Text**: High-cardinality text features (captions, cover text) need more sophisticated handling than direct categorical mapping (e.g., TF-IDF, embeddings) to be effective and avoid an explosion in feature dimensionality.
*   **Hyperparameter Optimization**: Systematic tuning of LightFM's hyperparameters (components, learning rate, regularization alphas) using a dedicated validation set.

## Conclusion

This project successfully established a pipeline for developing and evaluating a recommender system on the KuaiRec dataset. Experiments highlighted the strong potential of using deep engagement signals like `watch_ratio_capped` for ranking. However, training a predictive LightFM model to effectively generalize and achieve high performance on unseen test data proved challenging, primarily due to overfitting and the difficulty of translating broad feature signals into precise, personalized recommendations that match specific future interactions.

The "play_count only" model's zero performance underscored that simple global popularity from the training period was insufficient. The oracle evaluation using actual `watch_ratio_capped` set a high benchmark, indicating that if a model *could* predict this deep engagement, the recommendations would be highly precise.

Future efforts should focus on robust feature engineering (especially for text and temporal signals), systematic hyperparameter tuning with a proper validation strategy, and stronger regularization to combat overfitting, aiming to bridge the gap between current model performance and the potential demonstrated by oracle ranking.
