__all__ = ['DataProcessor', 'SingleImageGenerator']

import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, DenseNet121, Xception
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
    array_to_img
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib as mat
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict
import gc
from collections import Counter
import logging
from tqdm import tqdm
import mlflow
import mlflow.tensorflow
from collections import Counter



# In data_processing/generators.py

class SingleImageGenerator:
    """Lightweight generator for single image prediction"""
    def __init__(self, preprocessed_image: np.ndarray):
        """
        Initialize with preprocessed image
        
        Args:
            preprocessed_image: Numpy array of shape (1, height, width, channels)
        """
        self.preprocessed_image = preprocessed_image
        self.batch_size = 1
        self.n = 1
        self.i = 0
        
    def __iter__(self):
        """Make the class iterable"""
        self.i = 0
        return self
    
    def __next__(self):
        """Get next batch"""
        if self.i < self.n:
            self.i += 1
            return self.preprocessed_image
        else:
            raise StopIteration()
            
    def reset(self):
        """Reset the generator"""
        self.i = 0

    def __len__(self):
        """Return the number of batches"""
        return self.n

    # Add Tensorflow Dataset compatibility method
    def as_numpy_iterator(self):
        """Make compatible with TensorFlow's prediction methods"""
        return iter([self.preprocessed_image])

class DataProcessor:
    def __init__(self, base_dir: str, img_size: Tuple[int, int] = (224, 224)):
        """
        Initialize DataProcessor with base directory and image size
        
        Args:
            base_dir: Base directory containing the dataset
            img_size: Tuple of (height, width) for input images
        """
        self.base_dir = base_dir
        self.img_size = img_size
        self.train_dir = os.path.join(base_dir, "train")
        self.valid_dir = os.path.join(base_dir, "valid")
        self.test_dir = os.path.join(base_dir, "test")
        
        # Initialize data generators
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        self.valid_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Logger setup
        self.logger = logging.getLogger(__name__)
        
        # Initialize class names
        self._initialize_class_names()

    def _initialize_class_names(self) -> None:
        try:
            train_path = os.path.join(self.base_dir, "train")
            if not os.path.exists(train_path):
                raise ValueError(f"Train directory not found at: {train_path}")
            
            # Get the disease classes directly from the train directory
            class_dirs = [d for d in os.listdir(train_path) 
                     if os.path.isdir(os.path.join(train_path, d))]
            
            if not class_dirs:
                raise ValueError("No class directories found in train directory")
            
            self.class_names = sorted(class_dirs)
            self.logger.info(f"Found {len(self.class_names)} classes")
        
        except Exception as e:
            self.logger.error(f"Error initializing class names: {str(e)}")
            raise

    def get_class_names(self) -> List[str]:
        """Get list of class names from the dataset directory"""
        if not hasattr(self, 'class_names') or not self.class_names:
            self._initialize_class_names()
        return self.class_names

    def process_single_image(self, image_path: str, apply_augmentation: bool = False) -> np.ndarray:
        """
        Process a single image for prediction
        
        Args:
            image_path: Path to the image file
            apply_augmentation: Whether to apply augmentation
            
        Returns:
            Preprocessed image array of shape (1, height, width, channels)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            img = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img)

            if img_array.size == 0:
                self.logger.error(f"Preprocessed image is empty for {image_path}")
                raise ValueError("Preprocessed image array is empty.")
        
            if apply_augmentation:
                img_array = np.expand_dims(img_array, 0)
                for batch in self.train_datagen.flow(img_array, batch_size=1):
                    img_array = batch
                    break
            else:
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, 0)

            if len(img_array.shape) != 4:
                self.logger.error(f"Invalid image array shape for {image_path}: {img_array.shape}")
                raise ValueError("Invalid image array shape. Expected (1, height, width, channels).")

            self.logger.info(f"Processed image shape: {img_array.shape}")
            return img_array

        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

    def get_single_image_generator(self, image_path: str, apply_augmentation: bool = False) -> SingleImageGenerator:
        """
        Create a lightweight generator for single image prediction
        
        Args:
            image_path: Path to the image file
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            SingleImageGenerator instance with preprocessed image
        """
        processed_image = self.process_single_image(image_path, apply_augmentation)
        return SingleImageGenerator(processed_image)

    def get_class_names(self) -> List[str]:
        """
        Get list of class names from the dataset directory
        
        Returns:
            List of class names
        """
        if not hasattr(self, 'class_names') or not self.class_names:
            self._initialize_class_names()
        return self.class_names

    def _get_file_paths(self) -> Dict[str, List]:
        """
        Get file paths and labels from the base directory
        
        Returns:
            Dictionary containing file paths and corresponding labels
        """
        image_paths = []
        labels = []
        
        for class_name in self.get_class_names():
            class_dir = os.path.join(self.base_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for image_name in os.listdir(class_dir):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, image_name))
                    labels.append(class_name)
        
        return {'paths': image_paths, 'labels': labels}

    def analyze_class_distribution(self, labels: List[str]) -> None:
        """
        Analyze and print the distribution of classes in the dataset
        
        Args:
            labels: List of class labels
        """
        distribution = Counter(labels)
        print("\nClass Distribution:")
        for label, count in distribution.items():
            print(f"{label}: {count}")

    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict:
        """
        Split dataset into train, validation, and test sets with stratification
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            
        Returns:
            Dictionary of class weights if successful, None if directories already exist
        """
        self.logger.info("Starting dataset split...")

        if os.path.exists(self.train_dir) and os.path.exists(self.valid_dir) and os.path.exists(self.test_dir):
            self.logger.info("Train, validation, and test directories already exist. Skipping dataset split.")
            return None

        self.logger.info("Splitting dataset into train, validation, and test sets...")
        data = self._get_file_paths()
        self.analyze_class_distribution(data['labels'])

        X_train, X_temp, y_train, y_temp = train_test_split(
            data['paths'],
            data['labels'],
            train_size=train_ratio,
            stratify=data['labels'],
            random_state=42
        )

        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp,
            y_temp,
            train_size=val_ratio_adjusted,
            stratify=y_temp,
            random_state=42
        )

        self.create_directory_structure()
        self.copy_files(X_train, y_train, self.train_dir)
        self.copy_files(X_valid, y_valid, self.valid_dir)
        self.copy_files(X_test, y_test, self.test_dir)

        class_weights = self._calculate_class_weights(y_train)
        return class_weights

    def plot_class_distribution(self, labels: List[str], title: str = "Class Distribution") -> None:
        """
        Plot the distribution of classes in the dataset
        
        Args:
            labels: List of class labels
            title: Title for the plot
        """
        distribution = Counter(labels)
        class_names = list(distribution.keys())
        class_counts = list(distribution.values())

        plt.figure(figsize=(12, 6))
        plt.bar(class_names, class_counts)
        plt.title(title)
        plt.xlabel("Classes")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def _calculate_class_weights(self, y_train: List[str]) -> Dict[int, float]:
        """
        Calculate class weights to handle class imbalance
        
        Args:
            y_train: List of training labels
            
        Returns:
            Dictionary of class weights
        """
        counter = Counter(y_train)
        max_samples = max(counter.values())
        return {i: max_samples / count for i, (label, count) in enumerate(counter.items())}

    def create_directory_structure(self) -> None:
        """Create directories for train, validation, and test sets"""
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.valid_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def copy_files(self, file_paths: List[str], labels: List[str], destination_dir: str) -> None:
        """
        Copy files to the specified directory, organized by class
        
        Args:
            file_paths: List of file paths
            labels: List of corresponding labels
            destination_dir: Destination directory
        """
        for path, label in zip(file_paths, labels):
            label_dir = os.path.join(destination_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            file_name = os.path.basename(path)
            destination_path = os.path.join(label_dir, file_name)
            if not os.path.exists(destination_path):
                os.link(path, destination_path)

    def create_data_generators(self) -> Tuple:
        """
        Create data generators for training, validation, and testing
        
        Returns:
            Tuple of (train_generator, valid_generator, test_generator)
        """
        self.logger.info("Creating data generators...")

        train_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=64,
            class_mode='categorical',
            shuffle=True
        )

        valid_generator = self.valid_test_datagen.flow_from_directory(
            self.valid_dir,
            target_size=self.img_size,
            batch_size=64,
            class_mode='categorical',
            shuffle=False
        )

        test_generator = self.valid_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=64,
            class_mode='categorical',
            shuffle=False
        )

        self.logger.info("Data generators created successfully")
        return train_generator, valid_generator, test_generator