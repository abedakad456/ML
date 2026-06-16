import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
def get_columns_to_drop():
    """Returns a list of columns to explicitly drop from the feature matrix."""
    return [
        # Forbidden / Raw text
        'task_uid', 'query', 'tool_names', 
        # Pipe-separated columns (excluded to prevent dimensionality explosion)
        'query_aspects', 'tool_aspects',
        # Multicollinearity (Dropping due to 0.84 correlation with num_available_tools)
        'total_params' 
    ]

def build_preprocessor(X_train: pd.DataFrame):
    """
    Builds and returns a fitted scikit-learn ColumnTransformer.
    """
    # 1. Identify column types based on what remains after dropping
    cols_to_drop = get_columns_to_drop()
    available_cols = [c for c in X_train.columns if c not in cols_to_drop]
    
    ordinal_cols = ['task_complexity']
    nominal_cols = ['task_domain']
    
    # Numeric columns are everything else
    numeric_cols = [c for c in available_cols if c not in ordinal_cols + nominal_cols]

    # 2. Define the individual transformation pipelines
    # Map complexity: low=1, medium=2, high=3
    ordinal_transformer = OrdinalEncoder(categories=[['low', 'medium', 'high']])
    
    # One-Hot Encode domains (ignore unknown domains if they pop up in the test set)
    nominal_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Scale numerical features for KNN
    numeric_transformer = StandardScaler()

    # 3. Combine into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', ordinal_transformer, ordinal_cols),
            ('nom', nominal_transformer, nominal_cols),
            ('num', numeric_transformer, numeric_cols)
        ],
        remainder='drop' # Explicitly drops any column not explicitly transformed (like our cols_to_drop)
    )

    return preprocessor, numeric_cols, nominal_cols, ordinal_cols

def missing_value_imputation(df_features: pd.DataFrame):
    # 80% train, 10% validation, 10% test using random_state=42.

    df_train, df_temp = train_test_split(
        df_features, 
        test_size=0.20, 
        stratify=df_features['can_answer'], 
        random_state=42
    )

    # Second, split the remaining 20% in half to get 10% val and 10% test
    df_val, df_test = train_test_split(
        df_temp, 
        test_size=0.50, 
        stratify=df_temp['can_answer'], 
        random_state=42
    )

    # will use the median to fill missing values, as it is robust to the long-tail outliers 
    # Discovered during data exploration.

    imputer = SimpleImputer(strategy='median')

    df_train['avg_param_description_length'] = imputer.fit_transform(df_train[['avg_param_description_length']])
    df_val['avg_param_description_length'] = imputer.transform(df_val[['avg_param_description_length']])
    df_test['avg_param_description_length'] = imputer.transform(df_test[['avg_param_description_length']])
    return df_train, df_val, df_test

def preprocess_data(df_features):
    df_train, df_val, df_test = missing_value_imputation(df_features)
    # 1. Separate Features (X) and Target (y)
    X_train = df_train.drop(columns=['can_answer'])
    y_train = df_train['can_answer']
    X_val = df_val.drop(columns=['can_answer'])
    y_val = df_val['can_answer']
    X_test = df_test.drop(columns=['can_answer'])
    y_test = df_test['can_answer']
    
    # 3. Build and FIT the preprocessor on the training set
    preprocessor, num_cols, nom_cols, ord_cols = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # 4. Transform all sets
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test