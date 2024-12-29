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