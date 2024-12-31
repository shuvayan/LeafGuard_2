import tensorflow as tf
import numpy as np
import os
from PIL import Image
import logging

from leafguard_capstone.data_processing.generators import DataProcessor
from leafguard_capstone.model_training.ensembles import VotingEnsemble, AveragingEnsemble
from leafguard_capstone.config.core import TRAINED_MODEL_DIR, DATASET_DIR
from leafguard_capstone.prediction_service.inference import ImagePredictor

class PredictionService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_configs = [
            {'name': 'mobilenetV2', 'file': 'base_model_1_mobilenetV2.h5'},
            {'name': 'densenet', 'file': 'base_model_2_densenet.h5'},
            {'name': 'xception', 'file': 'base_model_3_xception.h5'}
        ]
        
        # Initialize ensembles
        self.voting_ensemble = VotingEnsemble(TRAINED_MODEL_DIR, self.model_configs)
        self.averaging_ensemble = AveragingEnsemble(TRAINED_MODEL_DIR, self.model_configs)
        
        # Initialize predictors as None - will be created when needed
        self.single_predictor = None
        self.dataset_predictor = None

    def predict(self, input_path: str, apply_augmentation: bool = False) -> dict:
        """Smart prediction function that handles both single images and dataset directories"""
        try:
            if os.path.isfile(input_path):
                self.logger.info("Single image prediction mode")
                return self._predict_single_image(input_path, apply_augmentation)
            elif os.path.isdir(input_path):
                self.logger.info("Dataset-based prediction mode")
                return self._predict_with_dataset(input_path)
            else:
                raise ValueError(f"Invalid input path: {input_path}")
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def _predict_single_image(self, image_path: str, apply_augmentation: bool) -> dict:
        """Handle single image prediction efficiently"""
        try:
            if self.single_predictor is None:
                # Initialize with None for test_gen as it's a single image
                self.single_predictor = ImagePredictor(
                    self.voting_ensemble,
                    self.averaging_ensemble,
                    test_gen=None
                )
            return self.single_predictor.predict_image(image_path)
        except Exception as e:
            self.logger.error(f"Single image prediction failed: {str(e)}")
            raise

    def _predict_with_dataset(self, dataset_path: str) -> dict:
        """Handle dataset-based prediction with metrics"""
        try:
            # Initialize data processor and create generators
            processor = DataProcessor(dataset_path)
            _, _, test_gen = processor.create_data_generators()
            
            # Initialize predictor if needed
            if self.dataset_predictor is None:
                self.dataset_predictor = ImagePredictor(
                    self.voting_ensemble,
                    self.averaging_ensemble,
                    test_gen=test_gen
                )
            
            # Get predictions and metrics
            predictions = self.dataset_predictor.predict_image(dataset_path)
            metrics = self._calculate_dataset_metrics(test_gen)
            
            return {
                'mode': 'dataset',
                'predictions': predictions,
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"Dataset prediction failed: {str(e)}")
            raise

    def _calculate_dataset_metrics(self, test_gen) -> dict:
        """Calculate metrics for dataset prediction"""
        voting_pred, voting_metrics = self.voting_ensemble.predict_and_evaluate(test_gen)
        avg_pred, avg_metrics = self.averaging_ensemble.predict_and_evaluate(test_gen)
        
        return {
            'voting_metrics': voting_metrics,
            'averaging_metrics': avg_metrics
        }

def make_prediction(input_path: str, apply_augmentation: bool = False) -> dict:
    """
    Convenience function for making predictions
    
    Args:
        input_path: Path to either a single image or dataset directory
        apply_augmentation: Whether to apply augmentation (for single image only)
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        service = PredictionService()
        return service.predict(input_path, apply_augmentation)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

# Command-line interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prediction.py <path_to_image_or_dataset> [--augment]")
        sys.exit(1)
        
    input_path = sys.argv[1]
    apply_augmentation = "--augment" in sys.argv
    
    try:
        results = make_prediction(input_path, apply_augmentation)
        
        # Print results based on mode
        if isinstance(results, dict) and results.get('mode') == 'dataset':
            print("\nDataset Prediction Results:")
            print("Voting Metrics:", results['metrics']['voting_metrics'])
            print("Averaging Metrics:", results['metrics']['averaging_metrics'])
        else:
            print("\nSingle Image Prediction Results:")
            print(f"Image: {results['image_path']}")
            print(f"Voting Prediction: {results['voting_prediction']}")
            print(f"Averaging Prediction: {results['averaging_prediction']}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)