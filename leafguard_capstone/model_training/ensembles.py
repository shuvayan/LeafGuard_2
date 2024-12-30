# -*- coding: utf-8 -*-

import os
from typing import Tuple, List, Dict
import tensorflow as tf
import mlflow,numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from matplotlib import pyplot as plt
from leafguard_capstone.data_processing.generators import DataPreprocessor


class VotingEnsemble:
    def __init__(self, model_dir: str, model_configs: List[Dict]):
        """
        Initialize VotingEnsemble with saved model configurations

        Args:
            model_dir: Directory containing saved models
            model_configs: List of dictionaries containing model names and files
        """
        self.model_dir = model_dir
        self.models = self._load_models(model_configs)

    def _load_models(self, model_configs: List[Dict]) -> List[tf.keras.Model]:
        """Load all saved base models"""
        loaded_models = []
        for config in model_configs:
            model_path = os.path.join(self.model_dir, config['file'])
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                loaded_models.append(model)
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        return loaded_models

    def predict_and_evaluate(self, data_generator, get_metrics=True):
        """
        Make predictions using majority voting and optionally evaluate metrics

        Args:
            data_generator: Data generator containing the test data
            get_metrics: Boolean to determine if metrics should be calculated

        Returns:
            If get_metrics=True: Tuple of (predictions, metrics dictionary)
            If get_metrics=False: predictions only
        """
        # Get predictions from all models
        predictions = []
        for i, model in enumerate(self.models):
            print(f"\nMaking predictions with Model {i+1}: {model.name}")
            pred = model.predict(data_generator)
            pred_classes = np.argmax(pred, axis=1)
            predictions.append(pred_classes)

        # Convert to array for easier manipulation
        predictions = np.array(predictions)

        # Perform majority voting
        majority_votes = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )

        if not get_metrics:
            return majority_votes

        # Calculate metrics
        true_labels = data_generator.labels
        metrics = {
            'accuracy': accuracy_score(true_labels, majority_votes),
            'f1': f1_score(true_labels, majority_votes, average='weighted'),
            'precision': precision_score(true_labels, majority_votes, average='weighted'),
            'recall': recall_score(true_labels, majority_votes, average='weighted')
        }

        return majority_votes, metrics

class AveragingEnsemble:
    def __init__(self, model_dir: str, model_configs: List[Dict]):
        """
        Initialize AveragingEnsemble with saved model configurations

        Args:
            model_dir: Directory containing saved models
            model_configs: List of dictionaries containing model names and files
        """
        self.model_dir = model_dir
        self.models = self._load_models(model_configs)

    def _load_models(self, model_configs: List[Dict]) -> List[tf.keras.Model]:
        """Load all saved base models"""
        loaded_models = []
        for config in model_configs:
            model_path = os.path.join(self.model_dir, config['file'])
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                loaded_models.append(model)
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        return loaded_models

    def predict_and_evaluate(self, data_generator, get_metrics=True):
        """
        Make predictions using averaged probabilities

        Args:
            data_generator: Data generator containing the test data
            get_metrics: Boolean to determine if metrics should be calculated

        Returns:
            If get_metrics=True: Tuple of (predictions, metrics dictionary)
            If get_metrics=False: predictions only
        """
        # Get probability predictions from all models
        all_predictions = []
        for i, model in enumerate(self.models):
            print(f"\nMaking predictions with Model {i+1}: {model.name}")
            pred = model.predict(data_generator)
            all_predictions.append(pred)

        # Average the probabilities
        averaged_probs = np.mean(all_predictions, axis=0)
        final_predictions = np.argmax(averaged_probs, axis=1)

        if not get_metrics:
            return final_predictions

        # Calculate metrics
        true_labels = data_generator.labels
        metrics = {
            'accuracy': accuracy_score(true_labels, final_predictions),
            'f1': f1_score(true_labels, final_predictions, average='weighted'),
            'precision': precision_score(true_labels, final_predictions, average='weighted'),
            'recall': recall_score(true_labels, final_predictions, average='weighted')
        }

        return final_predictions, metrics

from pyngrok import ngrok

def compare_ensembles_with_mlflow(voting_ensemble, averaging_ensemble, test_gen, experiment_name="Ensemble_Comparison"):
    """
    Compare voting and averaging ensembles using MLflow for tracking

    Args:
        voting_ensemble: Initialized VotingEnsemble model
        averaging_ensemble: Initialized AveragingEnsemble model
        test_gen: Test data generator
        experiment_name: Name for the MLflow experiment
    """
    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Dictionary to store results for plotting
    all_metrics = {}

    with mlflow.start_run(run_name="ensemble_comparison"):
        #get_ipython().system_raw("mlflow ui --port 5000 &")
        ngrok.kill()
        NGROK_AUTH_TOKEN = "2qhVc92sVQKidlX23fU3woWGE9C_2V5NeudEYzXiPzmNVb1gf"
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        # Evaluate Voting Ensemble
        print("\nEvaluating Voting Ensemble...")
        voting_predictions, voting_metrics = voting_ensemble.predict_and_evaluate(test_gen)

        # Evaluate Averaging Ensemble
        print("\nEvaluating Averaging Ensemble...")
        avg_predictions, avg_metrics = averaging_ensemble.predict_and_evaluate(test_gen)

        # Log all metrics to MLflow in a single run
        for metric_name, value in voting_metrics.items():
            mlflow.log_metric(f"voting_{metric_name}", value)
            mlflow.log_metric(f"averaging_{metric_name}", avg_metrics[metric_name])

        # Store metrics for plotting
        all_metrics["Voting Ensemble"] = voting_metrics
        all_metrics["Averaging Ensemble"] = avg_metrics

        # Log comparison parameters
        mlflow.log_param("num_models", len(voting_ensemble.models))
        mlflow.log_param("model_architectures", [model.name for model in voting_ensemble.models])
        # Open an HTTPs tunnel on port 5000 for http://localhost:5000
        ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
        print("MLflow Tracking UI:", ngrok_tunnel.public_url)

    # Plot comparison
    plot_ensemble_comparison(all_metrics)

def plot_ensemble_comparison(metrics_dict):
    """
    Plot comparison of ensemble methods

    Args:
        metrics_dict: Dictionary containing metrics for each ensemble method
    """
    methods = list(metrics_dict.keys())
    metric_names = list(metrics_dict[methods[0]].keys())

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, method in enumerate(methods):
        values = [metrics_dict[method][metric] for metric in metric_names]
        ax.bar(x + i * width, values, width, label=method)

    ax.set_ylabel('Score')
    ax.set_title('Ensemble Methods Comparison')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_names)
    ax.legend()

    # Add value labels on top of bars
    for i, method in enumerate(methods):
        values = [metrics_dict[method][metric] for metric in metric_names]
        for j, v in enumerate(values):
            ax.text(j + i * width, v, f'{v:.3f}',
                   ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

