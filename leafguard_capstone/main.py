
import argparse
import sys
import os

from leafguard_capstone.model_training.ensembles import compare_ensembles_with_mlflow, VotingEnsemble, AveragingEnsemble
from leafguard_capstone.data_processing.generators import DataPreprocessor
from leafguard_capstone.prediction_service.inference import *
from leafguard_capstone.config.core import TRAINED_MODEL_DIR,DATASET_DIR

from pathlib import Path


if __name__ == 'main':
    # Make prediction
    image_path = DATASET_DIR + "/Working_Images/crn.jpg"
    try:
        results = make_prediction(image_path)
        print("\nPrediction Results:")
        print(f"Image: {results['image_path']}")
        print(f"Voting Ensemble Prediction: {results['voting_prediction']}")
        print(f"Averaging Ensemble Prediction: {results['averaging_prediction']}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

#python main.py --task predict --image /Users/sangeetadutta/Downloads/IISC/leafguard_capstone/datasets/Working_Images/crn.jpg --models_dir /Users/sangeetadutta/Downloads/IISC/leafguard_capstone/saved_models

