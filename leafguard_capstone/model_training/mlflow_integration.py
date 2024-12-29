from pyngrok import ngrok
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, DenseNet121, Xception
#from tensorflow.keras.applications import EfficientNetB0, DenseNet121, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import gc
from collections import Counter
import logging
from tqdm import tqdm
import mlflow
import mlflow.tensorflow

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
        get_ipython().system_raw("mlflow ui --port 5000 &")
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