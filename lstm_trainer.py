import numpy as np
import pandas as pd
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.model_selection import train_test_split
import joblib

# Configuration
MAX_VOCAB = 10000
MAX_LEN = 50
EMBEDDING_DIM = 128
LSTM_UNITS = 64

def generate_data(num_samples=5000):
    data = []
    for _ in range(num_samples):
        if np.random.rand() < 0.6:  # Real profiles
            data.append({
                'bio': ' '.join([
                    np.random.choice(['travel', 'photography', 'technology', 'fitness']),
                    'enthusiast',
                    'from',
                    np.random.choice(['new york', 'london', 'tokyo'])
                ]),
                'followers': np.random.randint(100, 5000),
                'following': np.random.randint(50, 500),
                'fake': 0
            })
        else:  # Fake profiles
            data.append({
                'bio': ' '.join([
                    'click',
                    np.random.choice(['free', 'limited', 'special']),
                    'offer',
                    'link in bio'
                ]),
                'followers': np.random.randint(0, 100),
                'following': np.random.randint(500, 5000),
                'fake': 1
            })
    return pd.DataFrame(data)

# Prepare data
df = generate_data()
texts = df['bio'].values
labels = df['fake'].values

# Text processing
tokenizer = Tokenizer(num_words=MAX_VOCAB)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X_text = pad_sequences(sequences, maxlen=MAX_LEN)

# Numerical features
X_num = df[['followers', 'following']].values
X_num = (X_num - X_num.mean(axis=0)) / X_num.std(axis=0)  # Normalize

# Split data
X_train_t, X_test_t, X_train_n, X_test_n, y_train, y_test = train_test_split(
    X_text, X_num, labels, test_size=0.2, random_state=42
)

# Build LSTM model
text_input = Input(shape=(MAX_LEN,), name='text_input')
embedding = Embedding(MAX_VOCAB, EMBEDDING_DIM)(text_input)
lstm_out = LSTM(LSTM_UNITS)(embedding)

num_input = Input(shape=(2,), name='num_input')
concat = Concatenate()([lstm_out, num_input])
dense = Dense(32, activation='relu')(concat)
output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[text_input, num_input], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(
    [X_train_t, X_train_n],
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=([X_test_t, X_test_n], y_test)
)

# Save artifacts
model.save('lstm_model.h5')
joblib.dump(tokenizer, 'tokenizer.pkl')