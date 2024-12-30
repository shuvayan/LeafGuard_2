
import argparse
import sys
import os

from leafguard_capstone.model_training.ensembles import compare_ensembles_with_mlflow, VotingEnsemble, AveragingEnsemble
from leafguard_capstone.data_processing.generators import DataPreprocessor
from leafguard_capstone.prediction_service.inference import *

from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run LeafGuard Capstone project.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["train_ensembles", "predict"],
        help="Specify the task to execute: train_ensembles or predict"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="/Users/sangeetadutta/Downloads/IISC/leafguard_capstone/datasets/Working_Images/crn.jpg",
        help="Path to the image for prediction (required for the predict task)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/Users/sangeetadutta/Downloads/IISC/leafguard_capstone/datasets",
        help="Path to the dataset directory (required for training)"
    )

    args = parser.parse_args()

    if args.task == "train_ensembles":
        print("Starting ensemble training and comparison...")

        # Initialize data preprocessor
        data_dir = args.dataset_dir
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        preprocessor = DataPreprocessor(data_dir)

        # Split dataset and get class weights
        preprocessor.split_dataset()
        preprocessor.plot_class_distribution(preprocessor._get_file_paths()['labels'],
                                             title="Class Distribution Before Augmentation")

        # Create data generators with augmentation
        train_gen, valid_gen, test_gen = preprocessor.create_data_generators()

        # Configure saved models
        model_configs = [
            {'name': 'mobilenetV2', 'file': 'leafguard_capstone/saved_models/base_model_1_mobilenetV2.h5'},
            {'name': 'densenet', 'file': 'leafguard_capstone/saved_models/base_model_2_densenet.h5'},
            {'name': 'xception', 'file': 'leafguard_capstone/saved_models/base_model_3_xception.h5'}
        ]

        # Initialize ensemble models
        voting_ensemble = VotingEnsemble('leafguard_capstone/saved_models', model_configs)
        averaging_ensemble = AveragingEnsemble('leafguard_capstone/saved_models', model_configs)

        # Compare ensembles
        compare_ensembles_with_mlflow(voting_ensemble, averaging_ensemble, test_gen)

    elif args.task == "predict":
        # Ensure image path is provided
        image_path = args.image_path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        print(f"Running prediction for image: {image_path}")

        # Configure saved models
        model_configs = [
            {'name': 'mobilenetV2', 'file': 'leafguard_capstone/saved_models/base_model_1_mobilenetV2.h5'},
            {'name': 'densenet', 'file': 'leafguard_capstone/saved_models/base_model_2_densenet.h5'},
            {'name': 'xception', 'file': 'leafguard_capstone/saved_models/base_model_3_xception.h5'}
        ]

        # Initialize ensemble models
        voting_ensemble = VotingEnsemble('leafguard_capstone/saved_models', model_configs)
        averaging_ensemble = AveragingEnsemble('leafguard_capstone/saved_models', model_configs)

        # Initialize predictor
        predictor = ImagePredictor(voting_ensemble, averaging_ensemble)

        # Run prediction
        result = make_prediction.predict_image(image_path)
        print(f"Prediction Results: {result}")
        return result


#if __name__ == "__main__":
#    main()

#python main.py --task predict --image /Users/sangeetadutta/Downloads/IISC/leafguard_capstone/datasets/Working_Images/crn.jpg --models_dir /Users/sangeetadutta/Downloads/IISC/leafguard_capstone/saved_models


if __name__ == 'main':
    # Make prediction
    try:
        results = main()
        print("\nPrediction Results:")
        print(f"Image: {results['image_path']}")
        print(f"Voting Ensemble Prediction: {results['voting_prediction']}")
        print(f"Averaging Ensemble Prediction: {results['averaging_prediction']}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")