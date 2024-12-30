
from leafguard_capstone.data_processing.generators import DataPreprocessor
from leafguard_capstone.model_training.ensembles import VotingEnsemble,AveragingEnsemble,compare_ensembles_with_mlflow,predictor
from leafguard_capstone.config.core import TRAINED_MODEL_DIR,DATASET_DIR

# Initialize data preprocessor
data_dir = DATASET_DIR +"/Plant_leave_diseases_dataset_without_augmentation/"  # Update with your dataset path
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
    {'name': 'mobilenetV2', 'file': TRAINED_MODEL_DIR +'/base_model_1_mobilenetV2.h5'},
    {'name': 'densenet', 'file': TRAINED_MODEL_DIR +'/base_model_2_densenet.h5'},
    {'name': 'xception', 'file': TRAINED_MODEL_DIR +'/base_model_3_xception.h5'}
]

# Initialize both ensemble methods
voting_ensemble = VotingEnsemble('saved_models', model_configs)
averaging_ensemble = AveragingEnsemble('saved_models', model_configs)

# Compare ensembles using MLflow
compare_ensembles_with_mlflow(voting_ensemble, averaging_ensemble, test_gen)

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
    
if __name__ == 'main':
    # Make prediction
    image_path = DATASET_DIR + "/Working_Images/crn.jpg"
    try:
        results = predictor.predict_image(image_path)
        print("\nPrediction Results:")
        print(f"Image: {results['image_path']}")
        print(f"Voting Ensemble Prediction: {results['voting_prediction']}")
        print(f"Averaging Ensemble Prediction: {results['averaging_prediction']}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")