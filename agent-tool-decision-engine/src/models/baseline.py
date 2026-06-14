from sklearn.dummy import DummyClassifier

def get_majority_class_baseline():
    """
    Returns a dummy classifier that always predicts the majority class.
    This serves as the foundational baseline for model evaluation.
    """
    return DummyClassifier(strategy="most_frequent", random_state=42)