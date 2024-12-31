import os
import logging
from typing import Tuple, List, Dict, Union, Any
import tensorflow as tf
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from matplotlib import pyplot as plt
from tqdm import tqdm
from pyngrok import ngrok

class BaseEnsemble:
    """Base class for ensemble models"""

    def __init__(self, model_dir: str, model_configs: List[Dict]):
        self.model_dir = model_dir
        self.models = self._load_models(model_configs)
        self.logger = logging.getLogger(self.__class__.__name__)  # Initialize logger
        self.logger.info("BaseEnsemble initialized successfully")

    def _load_models(self, model_configs: List[Dict]) -> List[tf.keras.Model]:
        """Load all saved base models"""
        loaded_models = []
        for config in tqdm(model_configs, desc="Loading models"):
            model_path = os.path.join(self.model_dir, config['file'])
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                loaded_models.append(model)
                self.logger.info(f"Loaded model: {config['name']}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        return loaded_models

    def _validate_generator(self, data_generator: Any) -> None:
        """Validate the data generator"""
        if not (hasattr(data_generator, '__iter__') or isinstance(data_generator, tf.data.Dataset)):
            raise ValueError("Generator must be iterable or tf.data.Dataset")

class VotingEnsemble(BaseEnsemble):
    def __init__(self, model_dir: str, model_configs: List[Dict]):
        super().__init__(model_dir, model_configs)  # Call BaseEnsemble's constructor
        self.logger.info("Voting Ensemble initialized successfully")

    def predict_and_evaluate(self, data_generator, get_metrics=True):
        """
        Make predictions using majority voting
        
        Args:
            data_generator: Data generator or SingleImageGenerator
            get_metrics: Boolean to determine if metrics should be calculated
        """
        self._validate_generator(data_generator)
        
        # Get predictions from all models
        predictions = []
        for i, model in enumerate(self.models):
            self.logger.info(f"Making predictions with Model {i+1}: {model.name}")
            
            try:
                pred = model.predict(data_generator, verbose=0)
                pred_classes = np.argmax(pred, axis=1)
                predictions.append(pred_classes)
            except Exception as e:
                self.logger.error(f"Error in model {i+1}: {str(e)}")
                raise

        predictions = np.array(predictions)
        
        # Perform majority voting
        majority_votes = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )

        if not get_metrics:
            return majority_votes

        if not hasattr(data_generator, 'labels') or data_generator.labels is None:
            self.logger.warning("No labels found in generator, skipping metrics calculation")
            return majority_votes

        metrics = self._calculate_metrics(data_generator.labels, majority_votes)
        return majority_votes, metrics

class AveragingEnsemble(BaseEnsemble):
    def __init__(self, model_dir: str, model_configs: List[Dict]):
        super().__init__(model_dir, model_configs)  # Call BaseEnsemble's constructor
        self.logger.info("Averaging Ensemble initialized successfully")

    def predict_and_evaluate(self, data_generator, get_metrics=True):
        """
        Make predictions using averaged probabilities
        
        Args:
            data_generator: Data generator or SingleImageGenerator
            get_metrics: Boolean to determine if metrics should be calculated
        """
        self._validate_generator(data_generator)
        
        # Get probability predictions from all models
        all_predictions = []
        for i, model in enumerate(self.models):
            self.logger.info(f"Making predictions with Model {i+1}: {model.name}")
            
            try:
                pred = model.predict(data_generator, verbose=0)
                all_predictions.append(pred)
            except Exception as e:
                self.logger.error(f"Error in model {i+1}: {str(e)}")
                raise

        # Average the probabilities
        averaged_probs = np.mean(all_predictions, axis=0)
        final_predictions = np.argmax(averaged_probs, axis=1)

        if not get_metrics:
            return final_predictions

        if not hasattr(data_generator, 'labels') or data_generator.labels is None:
            self.logger.warning("No labels found in generator, skipping metrics calculation")
            return final_predictions

        metrics = self._calculate_metrics(data_generator.labels, final_predictions)
        return final_predictions, metrics

    def _calculate_metrics(self, true_labels, predictions):
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(true_labels, predictions),
            'f1': f1_score(true_labels, predictions, average='weighted'),
            'precision': precision_score(true_labels, predictions, average='weighted'),
            'recall': recall_score(true_labels, predictions, average='weighted')
        }

def compare_ensembles_with_mlflow(
    voting_ensemble: VotingEnsemble,
    averaging_ensemble: AveragingEnsemble,
    test_gen,
    experiment_name: str = "Ensemble_Comparison"
):
    """Compare voting and averaging ensembles using MLflow"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting ensemble comparison experiment: {experiment_name}")

    mlflow.set_experiment(experiment_name)
    all_metrics = {}

    with mlflow.start_run(run_name="ensemble_comparison"):
        try:
            # Configure ngrok
            ngrok.kill()
            NGROK_AUTH_TOKEN = "2qhVc92sVQKidlX23fU3woWGE9C_2V5NeudEYzXiPzmNVb1gf"
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)

            # Evaluate ensembles
            logger.info("Evaluating Voting Ensemble...")
            with tqdm(total=1, desc="Voting Ensemble") as pbar:
                voting_predictions, voting_metrics = voting_ensemble.predict_and_evaluate(test_gen)
                pbar.update(1)

            logger.info("Evaluating Averaging Ensemble...")
            with tqdm(total=1, desc="Averaging Ensemble") as pbar:
                avg_predictions, avg_metrics = averaging_ensemble.predict_and_evaluate(test_gen)
                pbar.update(1)

            # Log metrics
            for metric_name, value in voting_metrics.items():
                mlflow.log_metric(f"voting_{metric_name}", value)
                mlflow.log_metric(f"averaging_{metric_name}", avg_metrics[metric_name])

            # Store metrics for plotting
            all_metrics["Voting Ensemble"] = voting_metrics
            all_metrics["Averaging Ensemble"] = avg_metrics

            # Log parameters
            mlflow.log_param("num_models", len(voting_ensemble.models))
            mlflow.log_param("model_architectures", 
                           [model.name for model in voting_ensemble.models])

            # Set up ngrok tunnel
            ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
            logger.info(f"MLflow Tracking UI: {ngrok_tunnel.public_url}")

        except Exception as e:
            logger.error(f"Error during ensemble comparison: {str(e)}")
            raise
        finally:
            # Plot comparison regardless of MLflow status
            plot_ensemble_comparison(all_metrics)

def plot_ensemble_comparison(metrics_dict: Dict):
    """Plot comparison of ensemble methods"""
    if not metrics_dict:
        return

    methods = list(metrics_dict.keys())
    metric_names = list(metrics_dict[methods[0]].keys())

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, method in enumerate(methods):
        values = [metrics_dict[method][metric] for metric in metric_names]
        bars = ax.bar(x + i * width, values, width, label=method)
        
        # Add value labels
        for j, v in enumerate(values):
            ax.text(j + i * width, v, f'{v:.3f}',
                   ha='center', va='bottom')

    ax.set_ylabel('Score')
    ax.set_title('Ensemble Methods Comparison')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_names)
    ax.legend()

    plt.tight_layout()
    plt.show()