import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class MetricsPlotter:
  @staticmethod
  def plot_training_metrics(histories: List[dict], model_names: List[str]):
      """Plot training and validation metrics for all models"""
      import re  # Add import at the top of the file if not already present

      # Define metric bases we want to plot
      metric_bases = ['loss', 'accuracy', 'precision', 'recall']

      # Create subplots
      fig, axes = plt.subplots(2, 2, figsize=(20, 15))
      axes = axes.ravel()

      # Plot each metric type
      for idx, metric_base in enumerate(metric_bases):
          for history, name in zip(histories, model_names):
            # Find the metrics that match our base (including numbered versions)
            train_metric = next((key for key in history.keys()
                                if metric_base in key.lower()
                                and not key.startswith('val')), None)
            val_metric = next((key for key in history.keys()
                              if metric_base in key.lower()
                              and key.startswith('val')), None)

            if train_metric:
              axes[idx].plot(history[train_metric], label=f'{name} (train)')
            if val_metric:
              axes[idx].plot(history[val_metric], label=f'{name} (val)')

            # Clean up metric name for display by removing any numeric suffixes
            display_metric = re.sub(r'_\d+$', '', metric_base).title()
            axes[idx].set_title(f'Model {display_metric}')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(display_metric)
            axes[idx].legend()

      plt.tight_layout()
      plt.show()

  @staticmethod
  def plot_classification_metrics(y_true, y_pred, model_name: str):
      """Plot classification metrics"""
      metrics = {
          'Accuracy': accuracy_score(y_true, y_pred),
          'F1 Score': f1_score(y_true, y_pred, average='weighted'),
          'Precision': precision_score(y_true, y_pred, average='weighted'),
          'Recall': recall_score(y_true, y_pred, average='weighted')
      }

      plt.figure(figsize=(10, 6))
      bars = plt.bar(metrics.keys(), metrics.values())
      plt.title(f'Classification Metrics - {model_name}')
      plt.ylim(0, 1)

      # Add value labels on top of bars
      for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

      plt.show()

      # Print metrics
      for metric, value in metrics.items():
        print(f"{model_name} - {metric}: {value:.3f}")
