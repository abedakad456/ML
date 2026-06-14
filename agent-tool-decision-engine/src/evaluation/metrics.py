from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification_model(y_true, y_pred, model_name="Model"):
    """
    Calculates and prints accuracy, precision, recall, and F1-score.
    Returns a dictionary of the metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    
    print(f"--- {model_name} Results ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}\n")
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Generates and displays a styled confusion matrix using Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    
    # 0 = Refuse (Negative), 1 = Answer (Positive)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Refuse (0)', 'Answer (1)'],
                yticklabels=['Refuse (0)', 'Answer (1)'])
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()