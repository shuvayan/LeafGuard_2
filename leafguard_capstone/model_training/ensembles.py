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

# Initialize data preprocessor
data_dir = "/Users/sangeetadutta/Downloads/IISC/LeafGuard_2/leafguard_capstone/datasets/Plant_leave_diseases_dataset_without_augmentation/"  # Update with your dataset path
preprocessor = DataPreprocessor(data_dir)
meta_epoch = 100

# Split dataset and get class weights
class_weights = preprocessor.split_dataset()
preprocessor.plot_class_distribution(preprocessor._get_file_paths()['labels'],
                                      title="Class Distribution Before Augmentation")

# Create data generators with Augmentation
train_gen, valid_gen, test_gen = preprocessor.create_data_generators()

# Configure saved models
model_configs = [
    {'name': 'mobilenetV2', 'file': '/Users/sangeetadutta/Downloads/IISC/LeafGuard_2/leafguard_capstone/saved_models/base_model_1_mobilenetV2.h5'},
    {'name': 'densenet', 'file': '/Users/sangeetadutta/Downloads/IISC/LeafGuard_2/leafguard_capstone/saved_models/base_model_2_densenet.h5'},
    {'name': 'xception', 'file': '/Users/sangeetadutta/Downloads/IISC/LeafGuard_2/leafguard_capstone/saved_models/base_model_3_xception.h5'}
]

# Initialize both ensemble methods
voting_ensemble = VotingEnsemble('saved_models', model_configs)
averaging_ensemble = AveragingEnsemble('saved_models', model_configs)

# Compare ensembles using MLflow
compare_ensembles_with_mlflow(voting_ensemble, averaging_ensemble, test_gen)

from PIL import Image
class ImagePredictor:
    def __init__(self, voting_ensemble, averaging_ensemble, img_size=(224, 224)):
        """
        Initialize predictor with both ensemble models

        Args:
            voting_ensemble: Trained VotingEnsemble model
            averaging_ensemble: Trained AveragingEnsemble model
            img_size: Tuple of (height, width) for input image size
        """
        self.voting_ensemble = voting_ensemble
        self.averaging_ensemble = averaging_ensemble
        self.img_size = img_size

        # Get class names from the test generator
        self.class_names = list(test_gen.class_indices.keys())

    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image tensor
        """
        # Load and resize image
        img = Image.open(image_path)
        img = img.resize(self.img_size)

        # Convert to array and preprocess
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0,1]

        return img_array

    def predict_image(self, image_path):
        """
        Make predictions using both ensemble methods

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing predictions from both methods
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Preprocess image
        img_array = self.preprocess_image(image_path)
        test_generator = tf.data.Dataset.from_tensor_slices(img_array).batch(1)

        # Get predictions from both ensembles
        voting_pred = self.voting_ensemble.predict_and_evaluate(
            test_generator, get_metrics=False)
        averaging_pred = self.averaging_ensemble.predict_and_evaluate(
            test_generator, get_metrics=False)

        # Get class labels
        voting_class = self.class_names[voting_pred[0]]
        averaging_class = self.class_names[averaging_pred[0]]

        return {
            'voting_prediction': voting_class,
            'averaging_prediction': averaging_class,
            'image_path': image_path
        }

predictor = ImagePredictor(voting_ensemble, averaging_ensemble)

