import tensorflow as tf
import numpy as np
import os
import logging
from typing import Dict, Optional, Union
from pathlib import Path
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
    array_to_img
)

from leafguard_capstone.data_processing.generators import DataProcessor
from leafguard_capstone.model_training.ensembles import VotingEnsemble, AveragingEnsemble, compare_ensembles_with_mlflow
from leafguard_capstone.config.core import TRAINED_MODEL_DIR, DATASET_DIR

class ImagePredictor:
    def __init__(self, voting_ensemble, averaging_ensemble, img_size=(224, 224), test_gen=None):
        self.voting_ensemble = voting_ensemble
        self.averaging_ensemble = averaging_ensemble
        self.img_size = img_size
        self.test_gen = test_gen
        self.class_names = list(test_gen.class_indices.keys()) if test_gen else None

    def preprocess_image(self, image_path):
        img = load_img.open(image_path)
        img = img.resize(self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    def predict_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img_array = self.preprocess_image(image_path)
        test_generator = tf.data.Dataset.from_tensor_slices(img_array).batch(1)

        voting_pred = self.voting_ensemble.predict_and_evaluate(
            test_generator, get_metrics=False)
        averaging_pred = self.averaging_ensemble.predict_and_evaluate(
            test_generator, get_metrics=False)

        voting_class = self.class_names[voting_pred[0]]
        averaging_class = self.class_names[averaging_pred[0]]

        return {
            'voting_prediction': voting_class,
            'averaging_prediction': averaging_class,
            'image_path': image_path
        }

class InferenceService:
    def __init__(self, model_configs: Optional[list] = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing InferenceService")
        
        self.model_configs = model_configs or [
            {'name': 'mobilenetV2', 'file': 'base_model_1_mobilenetV2.h5'},
            {'name': 'densenet', 'file': 'base_model_2_densenet.h5'},
            {'name': 'xception', 'file': 'base_model_3_xception.h5'}
        ]
        
        self._initialize_components()
        self.image_predictor = None

    def _initialize_components(self) -> None:
        try:
            data_dir = os.path.join(DATASET_DIR, "Plant_leave_diseases_dataset_without_augmentation")
            self.processor = DataProcessor(data_dir)
            
            self.voting_ensemble = VotingEnsemble(TRAINED_MODEL_DIR, self.model_configs)
            self.averaging_ensemble = AveragingEnsemble(TRAINED_MODEL_DIR, self.model_configs)
            
            self.logger.info("Successfully initialized all components")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise

    def predict(self, image_path: str, mode: str = 'predict', apply_augmentation: bool = False) -> Dict:
        """
        Unified prediction method that handles both single image and dataset modes
        
        Args:
            image_path: Path to the image file
            mode: 'predict' for single image or 'retrain' for dataset processing
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            if mode == 'predict':
                return self.predict_single_image(image_path, apply_augmentation)
            elif mode == 'retrain':
                self.initialize_dataset_processing()
                if self.image_predictor is None:
                    self.image_predictor = ImagePredictor(
                        self.voting_ensemble,
                        self.averaging_ensemble,
                        test_gen=self.test_generator
                    )
                return self.image_predictor.predict_image(image_path)
            else:
                raise ValueError(f"Invalid mode: {mode}. Use 'predict' or 'retrain'")
                
        except Exception as e:
            self.logger.error(f"Prediction failed in {mode} mode: {str(e)}")
            raise

    def predict_single_image(self, image_path: str, apply_augmentation: bool = False) -> Dict:
        """Efficient single image prediction without dataset processing"""
        try:
            self.logger.info(f"Processing single image: {image_path}")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image_generator = self.processor.get_single_image_generator(
                image_path, 
                apply_augmentation=apply_augmentation
            )
            
            voting_pred = self.voting_ensemble.predict_and_evaluate(
                image_generator, 
                get_metrics=False
            )
            
            averaging_pred = self.averaging_ensemble.predict_and_evaluate(
                image_generator, 
                get_metrics=False
            )
            
            class_names = self.processor.get_class_names()
            
            results = {
                'image_path': image_path,
                'voting_prediction': class_names[voting_pred[0]],
                'averaging_prediction': class_names[averaging_pred[0]],
                'preprocessing': {
                    'augmentation_applied': apply_augmentation,
                    'mode': 'single_image'
                }
            }
            
            self.logger.info("Single image prediction completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during single image prediction: {str(e)}")
            raise

    def initialize_dataset_processing(self) -> None:
        """Initialize full dataset processing"""
        try:
            self.logger.info("Initializing dataset processing")
            class_weights = self.processor.split_dataset()
            train_gen, valid_gen, test_gen = self.processor.create_data_generators()
            self.test_generator = test_gen
            self.logger.info("Dataset processing initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing dataset processing: {str(e)}")
            raise

# Create a global instance of the inference service
inference_service = InferenceService()

def make_prediction(image_path: str, mode: str = 'predict', apply_augmentation: bool = False) -> Dict:
    """
    Convenience function for making predictions
    
    Args:
        image_path: Path to the image file
        mode: 'predict' for single image or 'retrain' for dataset processing
        apply_augmentation: Whether to apply data augmentation
        
    Returns:
        Dictionary containing prediction results
    """
    return inference_service.predict(image_path, mode, apply_augmentation)