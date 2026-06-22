import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrices(models_dict, X_test, y_test):
    """
    Plots confusion matrices for multiple models in a grid.
    models_dict format: {'Model Name': trained_model}
    """
    num_models = len(models_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(5.5 * num_models, 8))
    
    if num_models == 1:
        axes = [axes]
        
    for ax, (name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    ax=ax,annot_kws={"size": 22, "weight": "bold"},
                    xticklabels=['Refuse (0)', 'Answer (1)'],
                    yticklabels=['Refuse (0)', 'Answer (1)'])
        ax.set_title(f'{name}\nConfusion Matrix', fontsize=16, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=20)
        
    plt.tight_layout()
    plt.show()
    sns.set_theme()

def plot_metric_comparison(metrics_df):
    """
    Creates a grouped bar chart comparing Accuracy, Precision, Recall, and F1 across models.
    metrics_df should have models as the index and metrics as columns.
    """
    # Reset index brings 'Model' back as a regular column, so we use it as the id_vars
    plot_data = metrics_df.reset_index().melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_data, x='Metric', y='Score', hue='Model')
    
    plt.title('Evaluation: Model Performance Comparison on Test Set')
    plt.ylim(0, 1.05) # Metrics go from 0 to 1
    plt.legend(loc='lower right')
    plt.show()