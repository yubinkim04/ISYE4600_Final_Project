import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
# Import load_model from Keras models
from tensorflow.keras.models import load_model
# Need pad_sequences again for preprocessing new text
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration (must match the saving script) ---
# Define the paths to your saved files
model_path_bidi = 'part2_bidierectional_model.h5'
model_path_vanilla = 'part2_vanilla_model.h5'
tokenizer_path = 'tokenizer.json'
embedding_matrix_path = 'embedding_matrix.npy' # Although the matrix is saved inside the model, you saved it separately too.

# Define the maximum sequence length used during training/padding
MAX_LEN = 500 # This must match the MAX_LEN used when you created and trained the models

# --- Loading the Components ---

# 1. Load the Keras Model(s)
print(f"Loading models...")
try:
    loaded_bidi_model = load_model(model_path_bidi)
    print(f"Successfully loaded bidirectional model from: {model_path_bidi}")
    # Optional: Print summary to verify
    # loaded_bidi_model.summary()
except Exception as e:
    print(f"Error loading bidirectional model {model_path_bidi}: {e}")
    loaded_bidi_model = None # Set to None if loading fails

try:
    loaded_vanilla_model = load_model(model_path_vanilla)
    print(f"Successfully loaded vanilla model from: {model_path_vanilla}")
    # Optional: Print summary to verify
    # loaded_vanilla_model.summary()
except Exception as e:
    print(f"Error loading vanilla model {model_path_vanilla}: {e}")
    loaded_vanilla_model = None # Set to None if loading fails


# 2. Load the Tokenizer
print(f"\nLoading tokenizer from: {tokenizer_path}")
try:
    with open(tokenizer_path, 'r') as f:
        tokenizer_json = json.load(f)
    loaded_tokenizer = tokenizer_from_json(tokenizer_json)
    print("Successfully loaded tokenizer.")
    # Optional: Check if it loaded correctly
    # print(f"Loaded tokenizer has {len(loaded_tokenizer.word_index)} words in index.")
except Exception as e:
    print(f"Error loading tokenizer from {tokenizer_path}: {e}")
    loaded_tokenizer = None # Set to None if loading fails


# 3. Load the Embedding Matrix (Optional, as it's already in the model)
# You only need to load this separately if you plan to use the matrix itself
# for something other than feeding it into the loaded model.
print(f"\nLoading embedding matrix from: {embedding_matrix_path}")
try:
    loaded_embedding_matrix = np.load(embedding_matrix_path)
    print("Successfully loaded embedding matrix.")
    # Optional: Check shape
    # print(f"Loaded embedding matrix shape: {loaded_embedding_matrix.shape}")
except Exception as e:
    print(f"Error loading embedding matrix from {embedding_matrix_path}: {e}")
    loaded_embedding_matrix = None # Set to None if loading fails


# --- Example Usage: Preprocessing new text and making predictions ---

if loaded_tokenizer and (loaded_bidi_model or loaded_vanilla_model): # Check if necessary components loaded
    print("\n--- Example Prediction ---")
    # Assume you have new text data you want to classify
    new_texts = [
        "This sentence seems to be written by a human.",
        "Predict whether this machine generated text is real or fake.",
        "Here is another piece of text.",
        "Test data goes here for inference."
    ]

    print(f"Preprocessing {len(new_texts)} new texts...")
    # Use the loaded tokenizer to convert texts to sequences
    new_sequences = loaded_tokenizer.texts_to_sequences(new_texts)

    # Use pad_sequences with the SAME max_len used during training
    new_X = pad_sequences(
        new_sequences,
        maxlen=MAX_LEN,
        padding='post',  # Must match training
        truncating='post' # Must match training
    )
    print("Preprocessing complete.")
    print(f"Shape of preprocessed input: {new_X.shape}")

    # Make predictions using one of the loaded models (e.g., the bidirectional one)
    if loaded_bidi_model:
        print("\nMaking predictions with the bidirectional model...")
        predictions_proba = loaded_bidi_model.predict(new_X)

        # Convert probabilities to 1 or 2 labels (as discussed previously)
        # Threshold at 0.5, convert to int (0/1), then add 1 (1/2)
        predicted_labels_1_2 = (predictions_proba >= 0.5).astype(int) + 1

        print("Prediction Probabilities (Positive Class):")
        print(predictions_proba.flatten()) # Flatten for cleaner printing

        print("\nPredicted Labels (1 or 2):")
        print(predicted_labels_1_2.flatten()) # Flatten for cleaner printing

    elif loaded_vanilla_model: # If bidirectional failed, try the vanilla model
         print("\nMaking predictions with the vanilla model...")
         predictions_proba = loaded_vanilla_model.predict(new_X)
         predicted_labels_1_2 = (predictions_proba >= 0.5).astype(int) + 1
         print("Prediction Probabilities (Positive Class):")
         print(predictions_proba.flatten())
         print("\nPredicted Labels (1 or 2):")
         print(predicted_labels_1_2.flatten())

else:
    print("\nCould not load necessary components (tokenizer or models). Cannot run prediction example.")