import numpy as np
import matplotlib.pyplot as plt

def visualize():
    # Data for each model
    metrics = ['MSE', 'R2', 'Precision', 'Recall', 'F1-Score']
    models = ['Linear Regression', 'Decision Trees', 'Random Forests', 'XGBoost']

    # Values for each metric for the models
    values = {
        'MSE': [2535888.92, 2921926.66, 2916951.98, 2540685.80],
        'R2': [0.044, 0.1446, 0.1428, 0.1443],
        'Precision': [0.5017, 0.5081, 0.5079, 0.5015],
        'Recall': [0.6355, 0.6348, 0.6332, 0.6040],
        'F1-Score': [0.6355, 0.5032, 0.5029, 0.4193]
    }

    # Number of models
    n_models = len(models)
    x = np.arange(n_models)  # The label locations

    # Plot each metric in a separate image
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(x, values[metric], width=0.5, color=['blue', 'orange', 'green', 'red'])
        
        # Add labels, title, and custom x-axis tick labels
        plt.xlabel('Models')
        plt.ylabel(metric)
        plt.title(f'Model Comparison for {metric}')
        plt.xticks(x, models)
        
        # Save the plot to a file
        plt.tight_layout()
        plt.savefig(f'{metric}_comparison.png')  # Save each figure as a separate image
        plt.show()  # Show the figure
