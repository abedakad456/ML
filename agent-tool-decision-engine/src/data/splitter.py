import pandas as pd
from sklearn.model_selection import train_test_split

def perform_stratified_split(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """
    Splits data into 80% Train, 10% Validation, and 10% Test.
    Stratifies based on the target variable 'y'.
    """
    # 1. Split off the 80% Training set
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )
    
    # 2. Split the remaining 20% evenly to get 10% Validation and 10% Test
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.50, stratify=y_train_full, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test