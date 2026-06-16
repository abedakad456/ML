from sklearn.dummy import DummyClassifier

def get_majority_class_baseline(random_state: int = 42) -> DummyClassifier:
    """
    Returns a dummy classifier that always predicts the majority class (0 or 1).
    Used as the absolute baseline to evaluate actual model performance.
    """
    return DummyClassifier(strategy="most_frequent", random_state=random_state)