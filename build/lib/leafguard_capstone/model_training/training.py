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

class BaseModel:
    def __init__(self, model_name: str, img_size: Tuple[int, int], num_classes: int):
        self.model_name = model_name
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = self._create_model()

    def _create_model(self):
        input_shape = (*self.img_size, 3)

        if self.model_name == 'mobilenetV2':
            base = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
        elif self.model_name == 'densenet':
            base = DenseNet121(include_top=False, input_shape=input_shape, weights='imagenet')
        elif self.model_name == 'xception':
            base = Xception(include_top=False, input_shape=input_shape, weights='imagenet')

        # Freeze base model layers
        base.trainable = False

        # Build model with functional API
        inputs = tf.keras.Input(shape=input_shape)
        x = base(inputs)
        x = layers.GlobalAveragePooling2D(name='global_avg_pooling')(x)
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        x = layers.Dense(512, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.5, name='dropout_1')(x)
        x = layers.Dense(256, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.5, name='dropout_2')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, train_gen, valid_gen, class_weights, epochs=67):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()]
        )

        history = self.model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3
                )
            ]
        )
        return history

    def predict(self, data_gen):
        return self.model.predict(data_gen)
    
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