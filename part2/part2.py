#fill code
import io
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import sys
sys.path.append('../dataloader')
from dataloader import daigtv2_loader
sys.path.append('../part1')

# -------- 1. Load your data --------
# Replace with your actual data-loading code.
path_to_folder = "~/OneDrive - Georgia Institute of Technology/GT - Spring 2025/ISYE 4600/Final Project/"

df = daigtv2_loader(path_to_folder)
texts = df['text'].values
labels = df['label'].values

# -------- 2. Tokenize and pad --------
MAX_VOCAB = 20000
MAX_LEN   = 500

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

word_index = tokenizer.word_index
vocab_size = min(MAX_VOCAB, len(word_index)) + 1

# -------- 3. Load pretrained FastText vectors --------
EMBEDDING_DIM = 300
embedding_matrix = np.random.normal(
    size=(vocab_size, EMBEDDING_DIM)
).astype(np.float32)

def build_embedding_matrix(
    vec_file: str,
    word_index: dict,
    vocab_size: int,
    embedding_dim: int
) -> np.ndarray:
    """
    Reads `vec_file` (FastText .vec) line by line,
    and fills rows in embedding_matrix for words in word_index.
    """
    with io.open(vec_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        header = fin.readline()  # e.g. "1000000 300"
        for line in fin:
            parts = line.rstrip().split(' ')
            word = parts[0]
            if word in word_index:
                idx = word_index[word]
                if idx < vocab_size:
                    vect = np.asarray(parts[1:], dtype=np.float32)
                    if vect.shape[0] == embedding_dim:
                        embedding_matrix[idx] = vect
    return embedding_matrix

# Point this to wherever you downloaded “wiki-news-300d-1M.vec”
VEC_FILE = 'wiki-news-300d-1M.vec'
embedding_matrix = build_embedding_matrix(
    VEC_FILE, word_index, vocab_size, EMBEDDING_DIM
)


# -------- 4. Define the Keras model --------
model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=False,    # freeze pre-trained vectors
    ),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------- 5. Train --------
history = model.fit(
    X,
    labels,
    batch_size=32,
    epochs=5,
    validation_split=0.1
)