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
    _models_cache = {}  # Class variable to cache models

    def __init__(self, model_dir: str, model_configs: List[Dict]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing BaseEnsemble")
        
        self.model_dir = model_dir
        self.model_configs = model_configs
        self.models = self._load_models()
        
    def _load_models(self) -> List[tf.keras.Model]:
        """Load all saved base models with caching"""
        loaded_models = []
        for config in tqdm(self.model_configs, desc="Loading models"):
            model_path = os.path.join(self.model_dir, config['file'])
            if os.path.exists(model_path):
                # Check cache first
                if model_path in BaseEnsemble._models_cache:
                    model = BaseEnsemble._models_cache[model_path]
                    self.logger.info(f"Loaded model from cache: {config['name']}")
                else:
                    model = tf.keras.models.load_model(model_path)
                    BaseEnsemble._models_cache[model_path] = model
                    self.logger.info(f"Loaded model: {config['name']}")
                loaded_models.append(model)
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        return loaded_models

    def _validate_generator(self, data_generator: Any) -> None:
        """Validate the data generator"""
        valid_types = (
            hasattr(data_generator, '__iter__'),
            isinstance(data_generator, tf.data.Dataset),
            isinstance(data_generator, tf.keras.preprocessing.image.DirectoryIterator),
            hasattr(data_generator, 'preprocessed_image')  # For SingleImageGenerator
        )
        if not any(valid_types):
            raise ValueError("Invalid generator type. Must be iterable, Dataset, DirectoryIterator, or SingleImageGenerator")

    def _get_prediction_data(self, data_generator: Any) -> np.ndarray:
        """Extract prediction data from different types of generators"""
        try:
            if hasattr(data_generator, 'preprocessed_image'):
                return data_generator.preprocessed_image
            elif isinstance(data_generator, tf.data.Dataset):
                for data in data_generator:
                    if isinstance(data, tuple):
                        return data[0]
                    return data
            elif isinstance(data_generator, tf.keras.preprocessing.image.DirectoryIterator):
                return next(data_generator)[0]
            else:
                return next(iter(data_generator))
        except Exception as e:
            self.logger.error(f"Error extracting prediction data: {str(e)}")
            raise

class VotingEnsemble(BaseEnsemble):
    def __init__(self, model_dir: str, model_configs: List[Dict]):
        super().__init__(model_dir, model_configs)
        self.logger.info("Voting Ensemble initialized successfully")

    def predict_and_evaluate(self, data_generator, get_metrics=True):
        """Make predictions using majority voting"""
        self._validate_generator(data_generator)
        
        try:
            # Get data in the right format
            prediction_data = self._get_prediction_data(data_generator)
            
            # Get predictions from all models
            predictions = []
            for i, model in enumerate(self.models):
                pred = model.predict(prediction_data, verbose=0)
                pred_classes = np.argmax(pred, axis=1)
                predictions.append(pred_classes)

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
                self.logger.debug("No labels found in generator, skipping metrics calculation")
                return majority_votes

            metrics = self._calculate_metrics(data_generator.labels, majority_votes)
            return majority_votes, metrics
            
        except Exception as e:
            self.logger.error(f"Error during voting ensemble prediction: {str(e)}")
            raise

    def _calculate_metrics(self, true_labels, predictions):
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(true_labels, predictions),
            'f1': f1_score(true_labels, predictions, average='weighted'),
            'precision': precision_score(true_labels, predictions, average='weighted'),
            'recall': recall_score(true_labels, predictions, average='weighted')
        }

class AveragingEnsemble(BaseEnsemble):
    def __init__(self, model_dir: str, model_configs: List[Dict]):
        super().__init__(model_dir, model_configs)
        self.logger.info("Averaging Ensemble initialized successfully")

    def predict_and_evaluate(self, data_generator, get_metrics=True):
        """Make predictions using averaged probabilities"""
        self._validate_generator(data_generator)
        
        try:
            # Get data in the right format
            prediction_data = self._get_prediction_data(data_generator)
            
            # Get probability predictions from all models
            all_predictions = []
            for i, model in enumerate(self.models):
                pred = model.predict(prediction_data, verbose=0)
                all_predictions.append(pred)

            # Average the probabilities
            averaged_probs = np.mean(all_predictions, axis=0)
            final_predictions = np.argmax(averaged_probs, axis=1)

            if not get_metrics:
                return final_predictions

            if not hasattr(data_generator, 'labels') or data_generator.labels is None:
                self.logger.debug("No labels found in generator, skipping metrics calculation")
                return final_predictions

            metrics = self._calculate_metrics(data_generator.labels, final_predictions)
            return final_predictions, metrics
            
        except Exception as e:
            self.logger.error(f"Error during averaging ensemble prediction: {str(e)}")
            raise

    def _calculate_metrics(self, true_labels, predictions):
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(true_labels, predictions),
            'f1': f1_score(true_labels, predictions, average='weighted'),
            'precision': precision_score(true_labels, predictions, average='weighted'),
            'recall': recall_score(true_labels, predictions, average='weighted')
        }