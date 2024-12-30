import tensorflow as tf
import numpy as np
import os

from leafguard_capstone.data_processing.generators import DataPreprocessor
from leafguard_capstone.model_training.ensembles import VotingEnsemble,AveragingEnsemble,compare_ensembles_with_mlflow
from leafguard_capstone.config.core import TRAINED_MODEL_DIR,DATASET_DIR


# Initialize data preprocessor
data_dir = "//Home/leafguard_capstone/datasets/Plant_leave_diseases_dataset_without_augmentation/"  # Update with your dataset path
preprocessor = DataPreprocessor(data_dir)
meta_epoch = 100

# Split dataset and get class weights
class_weights = preprocessor.split_dataset()
preprocessor.plot_class_distribution(preprocessor._get_file_paths()['labels'],
                                      title="Class Distribution Before Augmentation")

# Create data generators with Augmentation
train_gen, valid_gen, test_gen = preprocessor.create_data_generators()

# Configure saved models
model_configs = [
    {'name': 'mobilenetV2', 'file': '//Home/leafguard_capstone/leafguard_capstone/saved_models/base_model_1_mobilenetV2.h5'},
    {'name': 'densenet', 'file': '//Home/leafguard_capstone/leafguard_capstone/saved_models/base_model_2_densenet.h5'},
    {'name': 'xception', 'file': '//Home/leafguard_capstone/leafguard_capstone/saved_models/base_model_3_xception.h5'}
]

# Initialize both ensemble methods
voting_ensemble = VotingEnsemble('saved_models', model_configs)
averaging_ensemble = AveragingEnsemble('saved_models', model_configs)

# Compare ensembles using MLflow
#compare_ensembles_with_mlflow(voting_ensemble, averaging_ensemble, test_gen)

from PIL import Image
class ImagePredictor:
    def __init__(self, voting_ensemble, averaging_ensemble, img_size=(224, 224)):
        """
        Initialize predictor with both ensemble models

        Args:
            voting_ensemble: Trained VotingEnsemble model
            averaging_ensemble: Trained AveragingEnsemble model
            img_size: Tuple of (height, width) for input image size
        """
        self.voting_ensemble = voting_ensemble
        self.averaging_ensemble = averaging_ensemble
        self.img_size = img_size

        # Get class names from the test generator
        self.class_names = list(test_gen.class_indices.keys())

    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image tensor
        """
        # Load and resize image
        img = Image.open(image_path)
        img = img.resize(self.img_size)

        # Convert to array and preprocess
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0,1]

        return img_array

    def predict_image(self, image_path):
        """
        Make predictions using both ensemble methods

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing predictions from both methods
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Preprocess image
        img_array = self.preprocess_image(image_path)
        test_generator = tf.data.Dataset.from_tensor_slices(img_array).batch(1)

        # Get predictions from both ensembles
        voting_pred = self.voting_ensemble.predict_and_evaluate(
            test_generator, get_metrics=False)
        averaging_pred = self.averaging_ensemble.predict_and_evaluate(
            test_generator, get_metrics=False)

        # Get class labels
        voting_class = self.class_names[voting_pred[0]]
        averaging_class = self.class_names[averaging_pred[0]]

        return {
            'voting_prediction': voting_class,
            'averaging_prediction': averaging_class,
            'image_path': image_path
        }

predictor = ImagePredictor(voting_ensemble, averaging_ensemble)


def make_prediction(image_path):
    """
    Function to make predictions on an image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        dict: Dictionary containing prediction results.
    """
    try:
        results = predictor.predict_image(image_path)
        return {
            "image_path": results['image_path'],
            "voting_prediction": results['voting_prediction'],
            "averaging_prediction": results['averaging_prediction'],
        }
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")
    
#if __name__ == 'main':
#    # Make prediction
#    image_path = DATASET_DIR + "/Working_Images/crn.jpg"
#    try:
#        results = make_prediction(image_path)
#        print("\nPrediction Results:")
#        print(f"Image: {results['image_path']}")
#        print(f"Voting Ensemble Prediction: {results['voting_prediction']}")
#        print(f"Averaging Ensemble Prediction: {results['averaging_prediction']}")
#    except Exception as e:
#        print(f"Error during prediction: {str(e)}")