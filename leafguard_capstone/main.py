#!/usr/bin/env python3

import argparse
import logging
import os
import time
import tensorflow as tf

from pathlib import Path
from leafguard_capstone.prediction_service.predictions import make_prediction


def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('LeafGuard')

def validate_path(path: str) -> str:
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Path does not exist: {path}")
    if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise argparse.ArgumentTypeError("File must be an image (PNG, JPG, JPEG)")
    return path

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='LeafGuard Plant Disease Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic prediction
    python main.py --input_path /path/to/image.jpg

    # With augmentation
    python main.py --input_path /path/to/image.jpg --augment

    # With verbose logging
    python main.py --input_path /path/to/image.jpg --verbose
        """
    )
    
    parser.add_argument(
        '--input_path', 
        type=validate_path,
        required=True,
        help='Path to the leaf image file'
    )
    
    parser.add_argument(
        '--augment', 
        action='store_true',
        help='Apply augmentation during prediction'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def format_prediction_output(results: dict) -> str:
    try:
        output = "\nPrediction Results:\n"
        output += "-" * 50 + "\n"
        output += f"Image: {results['image_path']}\n"
        output += f"Voting Ensemble Prediction: {results['voting_prediction']}\n"
        output += f"Averaging Ensemble Prediction: {results['averaging_prediction']}\n"
            
        if 'preprocessing' in results:
            output += f"\nPreprocessing Details:\n"
            output += f"Augmentation Applied: {results['preprocessing']['augmentation_applied']}\n"
            output += f"Mode: {results['preprocessing']['mode']}\n"
                    
        return output
    except Exception as e:
        return f"Error formatting output: {str(e)}\nRaw results: {results}"

def main():
    start_time = time.time()
    try:
        print("\n🌿 Starting LeafGuard Plant Disease Prediction...")
        args = parse_arguments()
        
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logger = setup_logging(log_level)
        
        print(f"📸 Analyzing image: {args.input_path}")
        print(f"🔄 Augmentation: {'enabled' if args.augment else 'disabled'}")
        
        print("\n⚙️  Processing...")
        results = make_prediction(args.input_path, args.augment)
        
        print("\n📊 Results:")
        print(format_prediction_output(results))
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Analysis complete! (Time: {elapsed_time:.2f}s)")
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

# Usage:
# python leafguard_capstone/main.py --input_path ~/Downloads/IISC/leafguard_capstone/datasets/Working_Images/crn.jpg --augment
# python leafguard_capstone/main.py --input_path ~/Downloads/IISC/leafguard_capstone/datasets/Working_Images/crn.jpg --augment --verbose