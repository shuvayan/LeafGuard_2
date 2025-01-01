#!/usr/bin/env python3

import argparse
import logging
import os
import time
import tensorflow as tf

from pathlib import Path
from leafguard_capstone.prediction_service.predictions import make_prediction
# Add these imports at the top of main.py
from transformers import pipeline
import torch
from leafguard_capstone.gradio_ui.llm_integration import apk

apk = apk

# def get_disease_treatment(disease_name):
#     """
#     Generate treatment information using local BLOOMZ model
#     """
#     try:
#         print("\n🤖 Loading local language model...")
#         generator = pipeline(
#             "text-generation",
#             model="bigscience/bloomz-560m",
#             device="cuda" if torch.cuda.is_available() else "cpu"
#         )
        
#         # More structured prompt with explicit formatting
#         prompt = f"""Task: Generate a plant disease analysis report.
# Disease Name: {disease_name}

# Please provide information in the following format:

# DESCRIPTION:
# [Brief overview of the disease and its impact on plants]

# DIAGNOSIS:
# - [Key symptom 1]
# - [Key symptom 2]
# - [Key symptom 3]

# TREATMENT:
# 1. Cultural Controls:
#    - [Method 1]
#    - [Method 2]

# 2. Chemical Controls:
#    - [Product 1]
#    - [Product 2]

# 3. Prevention:
#    - [Strategy 1]
#    - [Strategy 2]

# END REPORT"""
        
#         print("\n📝 Generating treatment information...")
#         response = generator(
#             prompt,
#             max_length=800,
#             min_length=200,
#             temperature=0.7,
#             top_p=0.9,
#             num_return_sequences=1,
#             do_sample=True,
#             truncation=True,
#             repetition_penalty=1.2
#         )[0]['generated_text']
        
#         # Clean up response
#         cleaned_response = response.split("Disease Name:")[1] if "Disease Name:" in response else response
#         cleaned_response = cleaned_response.replace(prompt, "").strip()
        
#         # Add formatting
#         if not cleaned_response.startswith("DESCRIPTION"):
#             cleaned_response = f"Analysis for {disease_name}:\n\n" + cleaned_response
            
#         return cleaned_response
        
#     except Exception as e:
#         print(f"\n❌ Error generating treatment: {str(e)}")
#         return f"Treatment information unavailable: {str(e)}"


import google.generativeai as genai

def get_disease_treatment(disease_name):
    """
    Generate treatment information using Google PaLM API
    """
    try:
        # Configure API
        genai.configure(api_key=apk)
        
        # Initialize model
        model = genai.GenerativeModel('gemini-pro')
        
        # Structured prompt
        prompt = f"""Generate a detailed plant disease analysis report for: {disease_name}

Please provide the following information:

DESCRIPTION:
[Provide brief overview of the disease]

DIAGNOSIS:
- [List main symptoms]
- [List identifying characteristics]
- [List potential damage indicators]

TREATMENT PLAN:
1. Cultural Controls:
   - [List recommended cultural practices]
   - [List preventive measures]

2. Chemical Controls:
   - [List appropriate fungicides/pesticides if applicable]
   - [List application timing and methods]

3. Prevention:
   - [List prevention strategies]
   - [List long-term management practices]"""

        # Generate response
        response = model.generate_content(prompt)
        
        # Check if we got a valid response
        if response.text:
            return response.text
        else:
            return "Unable to generate treatment information. Please try again."
            
    except Exception as e:
        print(f"\n❌ Error generating treatment: {str(e)}")
        return f"Treatment information unavailable: {str(e)}"
    

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
    
    # Launch UI
    python main.py --ui
        """
    )
    
    # Make input_path optional when using UI
    parser.add_argument(
        '--input_path', 
        type=validate_path,
        required=False,  # Changed from True
        help='Path to the leaf image file'
    )
    
    parser.add_argument(
        '--augment', 
        action='store_true',
        required=False,
        help='Apply augmentation during prediction'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Add UI argument
    parser.add_argument(
        '--ui',
        action='store_true',
        help='Launch the web interface'
    )
    
    return parser.parse_args()

DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___healthy', 'Cherry___Powdery_mildew',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___healthy',
    'Corn___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

#def get_disease_treatment(disease_name):
#    anthropic = Anthropic(api_key='')
#    prompt = f"Generate a complete description, diagnosis and treatment plan for: {disease_name}"
#    
#    try:
#        message = anthropic.messages.create(
#            model="claude-3-sonnet-20240229",
#            max_tokens=1000,
#            temperature=0,
#            messages=[{"role": "user", "content": prompt}]
#        )
#        return message.content
#    except Exception as e:
#        return f"Treatment information unavailable: {str(e)}"

def format_prediction_output(results: dict) -> str:
    try:
        output = "\nPrediction Results:\n"
        output += "-" * 50 + "\n"
        output += f"Image: {results['image_path']}\n"
        output += f"Voting Ensemble Prediction: {results['voting_prediction']}\n"
        output += f"Averaging Ensemble Prediction: {results['averaging_prediction']}\n"
        
        if 'treatment_info' in results:
            output += "\nTreatment Information:\n"
            output += "-" * 50 + "\n"
            output += results['treatment_info']
        
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

        # Handle UI mode
        if args.ui:
            from leafguard_capstone.gradio_ui.main import launch_ui
            print("\n🚀 Launching web interface...")
            launch_ui()
            return 0
            
        # Validate input path for CLI mode
        if not args.input_path:
            print("\n❌ Error: --input_path is required when not using --ui")
            return 1
        
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logger = setup_logging(log_level)
        
        print(f"📸 Analyzing image: {args.input_path}")
        print(f"🔄 Augmentation: {'enabled' if args.augment else 'disabled'}")
        
        print("\n⚙️  Processing...")
        results = make_prediction(args.input_path, args.augment)
        
        disease_name = results['voting_prediction']
        if disease_name in DISEASE_CLASSES and 'healthy' not in disease_name.lower():
            results['treatment_info'] = get_disease_treatment(disease_name)
        
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
# python leafguard_capstone/main.py --input_path ~/Downloads/IISC/leafguard_capstone/datasets/Working_Images/glb.jpeg --augment --verbose >> output.txt 2>&1
# python leafguard_capstone/main.py --ui
