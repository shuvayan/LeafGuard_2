import os
import logging
from typing import Dict, Optional
import tensorflow as tf
from pathlib import Path

from leafguard_capstone.data_processing.generators import DataProcessor, SingleImageGenerator
from leafguard_capstone.model_training.ensembles import VotingEnsemble, AveragingEnsemble
from leafguard_capstone.config.core import TRAINED_MODEL_DIR, DATASET_DIR

class ImagePredictor:
    def __init__(self, voting_ensemble, averaging_ensemble, img_size=(224, 224), test_gen=None):
        self.voting_ensemble = voting_ensemble
        self.averaging_ensemble = averaging_ensemble
        self.img_size = img_size
        self.test_gen = test_gen
        self.class_names = None
        self.logger = logging.getLogger(__name__)

    def preprocess_image(self, image_path, apply_augmentation=False):
        try:
            img_array = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img_array)
            img_array = img_array / 255.0
            img_array = tf.expand_dims(img_array, 0)
            return img_array
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {e}")
            raise

    def predict_image(self, image_path, processor=None):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            img_array = self.preprocess_image(image_path)
            img_gen = SingleImageGenerator(img_array)
            
            voting_pred = self.voting_ensemble.predict_and_evaluate(img_gen, get_metrics=False)
            averaging_pred = self.averaging_ensemble.predict_and_evaluate(img_gen, get_metrics=False)

            if processor:
                self.class_names = processor.get_class_names()
            elif not self.class_names and self.test_gen:
                self.class_names = list(self.test_gen.class_indices.keys())

            if not self.class_names:
                raise ValueError("No class names available")

            return {
                'voting_prediction': self.class_names[voting_pred[0]],
                'averaging_prediction': self.class_names[averaging_pred[0]],
                'image_path': image_path
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting image {image_path}: {str(e)}")
            raise

class PredictionService:
    _instance = None
    _models_cache = {}

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self.model_configs = [
                {'name': 'mobilenetV2', 'file': 'base_model_1_mobilenetV2.h5'},
                {'name': 'densenet', 'file': 'base_model_2_densenet.h5'},
                {'name': 'xception', 'file': 'base_model_3_xception.h5'}
            ]
            self._initialize_components()
            self._initialized = True

    def _initialize_components(self):
        try:
            self.processor = DataProcessor(DATASET_DIR)
            self.voting_ensemble = VotingEnsemble(TRAINED_MODEL_DIR, self.model_configs)
            self.averaging_ensemble = AveragingEnsemble(TRAINED_MODEL_DIR, self.model_configs)
            self.single_predictor = None
            self.logger.info("Components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise

    def predict(self, image_path: str, apply_augmentation: bool = False) -> Dict:
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"File not found: {image_path}")

            if self.single_predictor is None:
                self.single_predictor = ImagePredictor(
                    self.voting_ensemble,
                    self.averaging_ensemble
                )

            return self.single_predictor.predict_image(image_path, processor=self.processor)
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

def make_prediction(image_path: str, apply_augmentation: bool = True) -> Dict:
    try:
        service = PredictionService()
        return service.predict(image_path, apply_augmentation)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")