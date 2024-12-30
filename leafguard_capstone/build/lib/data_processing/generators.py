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
import matplotlib as mat
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict
import gc
from collections import Counter
import logging
from tqdm import tqdm
import mlflow
import mlflow.tensorflow

class DataPreprocessor:
    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (224, 224)):
      self.data_dir = Path(data_dir)
      self.img_size = img_size
      self.train_dir = Path('processed_data/train')
      self.valid_dir = Path('processed_data/valid')
      self.test_dir = Path('processed_data/test')

    def _get_file_paths(self) -> Dict[str, List[str]]:
      """Get all image paths and their corresponding labels"""
      image_paths = []
      labels = []
      extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG']

      for class_dir in self.data_dir.iterdir():
          if class_dir.is_dir():
              for ext in extensions:
                  for img_path in class_dir.glob(ext):
                      image_paths.append(str(img_path))
                      labels.append(class_dir.name)

      return {'paths': image_paths, 'labels': labels}

    def analyze_class_distribution(self, labels: List[str]) -> Dict[str, int]:
      """Analyze and print class distribution"""
      distribution = Counter(labels)
      print("Class distribution:")
      for class_name, count in distribution.items():
          print(f"{class_name}: {count}")
      return distribution

    def create_directory_structure(self):
      """Create directory structure for train/valid/test splits"""
      for dir_path in [self.train_dir, self.valid_dir, self.test_dir]:
          if dir_path.exists():
              shutil.rmtree(dir_path)
          dir_path.mkdir(parents=True)

      data = self._get_file_paths()
      unique_labels = set(data['labels'])

      for split_dir in [self.train_dir, self.valid_dir, self.test_dir]:
          for label in unique_labels:
              (split_dir / label).mkdir(parents=True, exist_ok=True)

    def copy_files(self, files: List[str], labels: List[str], dest_base_dir: Path):
      """Copy files to destination directory maintaining structure"""
      for file_path, label in tqdm(zip(files, labels), desc=f"Copying files to {dest_base_dir.name}"):
          dest_dir = dest_base_dir / label
          dest_file = dest_dir / Path(file_path).name
          shutil.copy2(file_path, dest_file)

    def split_dataset(self, train_ratio=0.7, val_ratio=0.15):
      """Split dataset into train, validation, and test sets with stratification"""
      data = self._get_file_paths()
      self.analyze_class_distribution(data['labels'])

      # Check for classes with only one sample and handle them:
      label_counts = Counter(data['labels'])
      # Create stratified train and temp splits
      X_train, X_temp, y_train, y_temp = train_test_split(
          data['paths'],
          data['labels'],
          train_size=train_ratio,
          stratify=data['labels'],
          random_state=42
      )

      # Split temp into validation and test
      val_ratio_adjusted = val_ratio / (1 - train_ratio)
      X_valid, X_test, y_valid, y_test = train_test_split(
          X_temp,
          y_temp,
          train_size=val_ratio_adjusted,
          stratify=y_temp,
          random_state=42
      )

      # Create directory structure and copy files
      self.create_directory_structure()
      self.copy_files(X_train, y_train, self.train_dir)
      self.copy_files(X_valid, y_valid, self.valid_dir)
      self.copy_files(X_test, y_test, self.test_dir)

      # Calculate class weights for handling imbalance
      class_weights = self._calculate_class_weights(y_train)
      return class_weights

    def plot_class_distribution(self, labels: List[str], title: str = "Class Distribution"):

      """Plots the distribution of classes in the dataset."""
      distribution = Counter(labels)
      class_names = list(distribution.keys())
      class_counts = list(distribution.values())

      plt.figure(figsize=(12, 6))
      plt.bar(class_names, class_counts)
      plt.title(title)
      plt.xlabel("Classes")
      plt.ylabel("Number of Samples")
      plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
      plt.tight_layout()
      plt.show()

    def _calculate_class_weights(self, y_train: List[str]) -> Dict[int, float]:
      """Calculate class weights to handle class imbalance"""
      counter = Counter(y_train)
      max_samples = max(counter.values())
      return {i: max_samples/count for i, (label, count) in
              enumerate(counter.items())}

    def create_data_generators(self):
      """Create data generators with augmentation for training"""
      train_datagen = ImageDataGenerator(
          rescale=1./255,  # Normalization
          rotation_range=20,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest'
      )

      valid_test_datagen = ImageDataGenerator(rescale=1./255)

      train_generator = train_datagen.flow_from_directory(
          self.train_dir,
          target_size=self.img_size,
          batch_size=64,
          class_mode='categorical',
          shuffle=True
      )

      valid_generator = valid_test_datagen.flow_from_directory(
          self.valid_dir,
          target_size=self.img_size,
          batch_size=64,
          class_mode='categorical',
          shuffle=False
      )

      test_generator = valid_test_datagen.flow_from_directory(
          self.test_dir,
          target_size=self.img_size,
          batch_size=64,
          class_mode='categorical',
          shuffle=False
      )

      return train_generator, valid_generator, test_generator

