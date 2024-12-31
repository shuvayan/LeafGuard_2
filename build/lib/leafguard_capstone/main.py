import argparse
import sys
import os
import logging
from pathlib import Path
from tqdm import tqdm
import time

from leafguard_capstone.model_training.ensembles import VotingEnsemble, AveragingEnsemble
from leafguard_capstone.data_processing.generators import DataProcessor
from leafguard_capstone.prediction_service.predictions import PredictionService, make_prediction
from leafguard_capstone.config.core import TRAINED_MODEL_DIR, DATASET_DIR

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('leafguard_predictions.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('LeafGuard')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Make predictions on plant disease images')
    
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to the image file or dataset directory')
    
    parser.add_argument('--augment', action='store_true',
                      help='Apply augmentation during prediction (only for single image mode)')
    
    parser.add_argument('--model_dir', type=str, default=TRAINED_MODEL_DIR,
                      help='Directory containing trained models')
    
    parser.add_argument('--log_level', type=str, 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      default='INFO',
                      help='Set the logging level')
    
    return parser.parse_args()

def display_progress(message):
    """Display progress bar with message"""
    with tqdm(total=100, desc=message) as pbar:
        for _ in range(10):
            time.sleep(0.1)
            pbar.update(10)

def main():
    """Main function to run predictions"""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(getattr(logging, args.log_level))
    logger.info("Starting LeafGuard prediction service...")
    
    try:
        # Log prediction attempt
        logger.info(f"Attempting prediction on: {args.input_path}")
        logger.info(f"Augmentation: {args.augment}")
        
        # Display progress for model loading
        display_progress("Loading models")
        
        # Make prediction with progress indication
        logger.info("Making prediction...")
        with tqdm(total=100, desc="Processing") as pbar:
            pbar.update(30)  # Loading
            
            results = make_prediction(
                input_path=args.input_path,
                apply_augmentation=args.augment
            )
            
            pbar.update(70)  # Prediction complete
        
        # Log and display results
        logger.info("Prediction completed successfully")
        
        print("\n" + "="*50)
        print("Prediction Results:")
        print("="*50)
        
        if isinstance(results, dict) and results.get('mode') == 'dataset':
            print("\nDataset Prediction Results:")
            print("Voting Metrics:", results['metrics']['voting_metrics'])
            print("Averaging Metrics:", results['metrics']['averaging_metrics'])
        else:
            print(f"Image: {results['image_path']}")
            print(f"Voting Ensemble Prediction: {results['voting_prediction']}")
            print(f"Averaging Ensemble Prediction: {results['averaging_prediction']}")
            
            if 'preprocessing' in results:
                print("\nPreprocessing Information:")
                print(f"Mode: {results['preprocessing']['mode']}")
                print(f"Augmentation Applied: {results['preprocessing']['augmentation_applied']}")
        
        print("="*50)
        
        logger.info("Results displayed successfully")
        
    except FileNotFoundError as e:
        error_msg = f"Error: Path not found - {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        sys.exit(1)
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(error_msg)
        sys.exit(1)
    
    logger.info("Prediction service completed successfully")

if __name__ == '__main__':
    main()