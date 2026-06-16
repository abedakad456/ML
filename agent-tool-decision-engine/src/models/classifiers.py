from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

def get_knn_model(**kwargs) -> KNeighborsClassifier:
    """Returns a K-Nearest Neighbors classifier."""
    return KNeighborsClassifier(**kwargs)

def get_adaboost_model(random_state: int = 42, **kwargs) -> AdaBoostClassifier:
    """Returns an AdaBoost classifier."""
    return AdaBoostClassifier(random_state=random_state, **kwargs)

def get_random_forest_model(random_state: int = 42, **kwargs) -> RandomForestClassifier:
    """Returns a Random Forest classifier (3rd supervised model)."""
    return RandomForestClassifier(random_state=random_state, **kwargs)