import tensorflow as tf
import numpy as np
import json
import os # Needed to check file existence or join paths
from transformers import BertTokenizer, TFBertModel
from catboost import CatBoostClassifier # Only Classifier is needed for loading/prediction
# pad_sequences is part of TensorFlow/Keras, need to import it
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys

# --- Define Paths ---
# These paths must match the paths used in your saving script
save_dir_transformer = './saved_transformer_model_tf'
save_dir_tokenizer = './saved_transformer_tokenizer_tf'
save_path_catboost = './saved_catboost_model_tf.cbm'
save_path_config = './model_config_tf.json'

# --- Load Configuration ---
print("Loading model configuration...")
try:
    with open(save_path_config, 'r') as f:
        config_data = json.load(f)

    # Get parameters and paths from the config file
    transformer_model_name = config_data.get('transformer_model_name')
    max_length = config_data.get('max_length')
    # It's safer to get the saved directories/paths from the config if you saved them there
    saved_transformer_model_dir = config_data.get('transformer_model_dir', save_dir_transformer) # Use default if not in config
    saved_transformer_tokenizer_dir = config_data.get('transformer_tokenizer_dir', save_dir_tokenizer) # Use default if not in config
    saved_catboost_model_path = config_data.get('catboost_model_path', save_path_catboost) # Use default if not in config


    if not all([transformer_model_name, max_length, saved_transformer_model_dir, saved_transformer_tokenizer_dir, saved_catboost_model_path]):
        raise ValueError("Missing required keys in config file.")

    print("Configuration loaded successfully.")
    print(f"Transformer Model Name: {transformer_model_name}")
    print(f"Max Sequence Length: {max_length}")
    print(f"Saved Transformer Model Dir: {saved_transformer_model_dir}")
    print(f"Saved Transformer Tokenizer Dir: {saved_transformer_tokenizer_dir}")
    print(f"Saved CatBoost Model Path: {saved_catboost_model_path}")


except FileNotFoundError:
     print(f"Error: Configuration file not found at {save_path_config}")
     sys.exit("Exiting: Could not find model configuration file.")
except json.JSONDecodeError:
     print(f"Error: Could not decode JSON from {save_path_config}")
     sys.exit("Exiting: Invalid model configuration file.")
except Exception as e:
    print(f"An unexpected error occurred loading config: {e}")
    sys.exit("Exiting: Error loading model configuration.")


# --- Load Tokenizer ---
print("\nLoading Transformer tokenizer...")
try:
    # Load from the directory where it was saved
    loaded_tokenizer = BertTokenizer.from_pretrained(saved_transformer_tokenizer_dir)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer from {saved_transformer_tokenizer_dir}: {e}")
    sys.exit("Exiting: Could not load tokenizer.")


# --- Load TensorFlow Transformer Model ---
print("\nLoading TensorFlow Transformer model...")
try:
    # Load from the directory where it was saved
    loaded_transformer_model = TFBertModel.from_pretrained(saved_transformer_model_dir)
    print("TensorFlow Transformer model loaded successfully.")
    # Ensure model is in evaluation mode (TensorFlow handles this usually)
    # loaded_transformer_model.trainable = False # Can set to False for inference
except Exception as e:
    print(f"Error loading Transformer model from {saved_transformer_model_dir}: {e}")
    sys.exit("Exiting: Could not load Transformer model.")


# --- Load CatBoost Model ---
print("\nLoading CatBoost model...")
try:
    loaded_catboost_model = CatBoostClassifier() # Instantiate the classifier
    loaded_catboost_model.load_model(saved_catboost_model_path) # Load the trained weights
    print("CatBoost model loaded successfully.")
except Exception as e:
    print(f"Error loading CatBoost model from {saved_catboost_model_path}: {e}")
    sys.exit("Exiting: Could not load CatBoost model.")


# --- Define Preprocessing Function ---
def preprocess_text(texts, tokenizer, max_length):
    """
    Tokenizes and pads/truncates texts to the specified max_length.
    Handles single strings or lists of strings.
    """
    if isinstance(texts, str):
        texts = [texts] # Ensure input is a list for batch processing

    # Tokenize the batch, returning TensorFlow tensors
    # Use the loaded tokenizer
    encoded_input = tokenizer(
        texts,
        padding=True,          # Pad sequences to the longest in the batch
        truncation=True,       # Truncate sequences longer than max_length
        max_length=max_length, # Maximum length of sequences
        return_tensors='tf'    # Return TensorFlow tensors
    )
    # Returns a dictionary with keys like 'input_ids', 'attention_mask', 'token_type_ids'
    return encoded_input

# --- Define Prediction Function ---
def predict_text_0_1(texts, tokenizer, transformer_model, catboost_model, max_length):
    """
    Predicts classification labels (0 or 1) for a list of raw texts.
    """
    # Preprocess the text using the loaded tokenizer and max_length
    encoded_input = preprocess_text(texts, tokenizer, max_length)

    # Get embeddings from the loaded Transformer model
    # Pass the dictionary of tensors to the model
    outputs = transformer_model(encoded_input)

    # Extract the [CLS] token embedding and convert to NumPy
    # outputs.last_hidden_state has shape (batch_size, sequence_length, hidden_size)
    # We take the first token ([CLS]) for all items in the batch: outputs.last_hidden_state[:, 0, :]
    cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    # Predict using the loaded CatBoost model
    # CatBoost's predict method directly returns the predicted class labels (0 or 1)
    predictions = catboost_model.predict(cls_embeddings)

    # Ensure the output is a 1D array if input was a list
    return predictions.flatten() if isinstance(texts, list) else predictions[0]


# --- Define Prediction Function (Output 1 or 2) ---
def predict_text_1_2(texts, tokenizer, transformer_model, catboost_model, max_length):
     """
     Predicts classification labels (1 or 2) for a list of raw texts.
     """
     # Get predictions as 0 or 1 first
     predictions_0_1 = predict_text_0_1(texts, tokenizer, transformer_model, catboost_model, max_length)
     # Convert 0/1 to 1/2 by adding 1
     predictions_1_2 = predictions_0_1 + 1
     return predictions_1_2


# --- Example Usage with New Text ---
# Check if all components were loaded successfully before attempting prediction
if loaded_tokenizer and loaded_transformer_model and loaded_catboost_model:
    print("\n--- Making Predictions on New Text ---")

    new_texts_to_classify = [
        "This is a brand new sentence for testing the loaded model.",
        "Another example text.",
        "Short sentence.",
        "A much longer piece of text that might get truncated depending on MAX_LEN."
    ]

    # Predict using the function that returns 0 or 1
    predictions_0_1 = predict_text_0_1(
        new_texts_to_classify,
        loaded_tokenizer,
        loaded_transformer_model,
        loaded_catboost_model,
        max_length # Use max_length loaded from config
    )
    print("\nPredictions (0 or 1):")
    print(predictions_0_1)

    # Predict using the function that returns 1 or 2
    predictions_1_2 = predict_text_1_2(
         new_texts_to_classify,
         loaded_tokenizer,
         loaded_transformer_model,
         loaded_catboost_model,
         max_length
    )
    print("\nPredictions (1 or 2):")
    print(predictions_1_2)

    # Example with a single string input
    single_text = "This is just one sentence."
    prediction_single_0_1 = predict_text_0_1(
        single_text,
        loaded_tokenizer,
        loaded_transformer_model,
        loaded_catboost_model,
        max_length
    )
    print(f"\nPrediction for single text '{single_text}' (0 or 1): {prediction_single_0_1}")

else:
    print("\nCould not load all model components. Cannot make predictions.")

