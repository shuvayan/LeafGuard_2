import os

class ModelSaver:
  def __init__(self, base_path="models"):
    self.base_path = base_path
    os.makedirs(self.base_path, exist_ok=True)

  def save_model(self, model, model_name, model_type="base", index=None):
    """Saves a model to disk.

    Args:
        model: The Keras model to save.
        model_name: The name of the model (e.g., 'mobilenetV2', 'meta_model').
        model_type: The type of model ('base' or 'meta'). Defaults to 'base'.
        index: The index of the base model (if applicable). Defaults to None.
    """
    if model_type == "base" and index is not None:
        model_path = os.path.join(self.base_path, f"base_model_{index + 1}_{model_name}.h5")
    elif model_type == "meta":
        model_path = os.path.join(self.base_path, f"{model_name}.h5")
    else:
        raise ValueError("Invalid model_type. Choose 'base' or 'meta'.")

    model.save(model_path)
    print(f"{model_type.capitalize()} model saved to: {model_path}")