import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def get_clustering_feature_names():
    """Returns the comprehensive list of features for clustering from all 4 groups."""
    return [
        # Group 1: Query/tool matching
        'aspect_coverage_ratio', 'aspect_overlap_count', 'aspect_mismatch_count', 
        'query_tool_token_jaccard', 'query_tool_action_overlap',
        
        # Group 2: Query complexity
        'query_word_count', 'query_multi_intent_score', 
        'query_specificity_score', 'query_unique_token_ratio',
        
        # Group 3: Tool complexity
        'num_available_tools', 'total_params', 'total_required_params', 
        'schema_rigidity_score', 'param_type_diversity',
        
        # Group 4: Risk signals
        'risky_tool_action_count', 'query_sensitive_data_signal', 
        'query_code_signal', 'query_temporal_signal'
    ]

def build_clustering_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts clustering features, imputes missing values, and scales the data.
    Returns a scaled Pandas DataFrame with column names preserved for interpretation.
    """
    features = get_clustering_feature_names()
    
    # Safely select only the features that actually exist in the dataframe
    # This prevents KeyErrors if a column name has a slight typo in the raw data
    available_features = [f for f in features if f in df.columns]
    X_cluster = df[available_features].copy()
    
    # 1. Impute any missing values (Median is robust against outliers)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_cluster)
    
    # 2. Scale the features (Mandatory for distance-based clustering algorithms)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 3. Reconstruct the DataFrame to preserve column names
    df_scaled = pd.DataFrame(X_scaled, columns=available_features, index=df.index)
    
    return df_scaled