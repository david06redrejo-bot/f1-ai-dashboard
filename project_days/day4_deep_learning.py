import os
# Suppress TensorFlow logs (Must be before importing tensorflow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- PART 1: The Dataset (Collecting the "Knowledge") ---
# Deep Learning needs examples to learn. Here we have F1 radio messages.
sentences = [
    # URGENT (1)
    "Box now, engine failing",
    "Stop the car immediately",
    "Fire at the back",
    "I have a puncture",
    "Brakes are not working",
    "Crash in sector two",
    "Heavy rain starting now",
    "Lose power, no drive",
    "Critical damage to floor",
    "Oil pressure dropping fast",
    
    # NORMAL (0)
    "The tires are slightly warm",
    "Keep pushing for two laps",
    "We are discussing the strategy",
    "Gap to leader is 5 seconds",
    "Battery charge is high",
    "DRS will be enabled",
    "Radio check, do you copy",
    "Understood, staying out",
    "Mode push when ready",
    "Tires feel good so far"
]

# Labels tell the computer the "Answer" for each example.
# 1 means "Urgent/Danger", 0 means "Normal/Info"
labels = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 10 Urgent
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0   # 10 Normal
])

# --- PART 2: Preprocessing (Translating English to Numbers) ---
# Computers don't understand words; they only understand math. 

# A Tokenizer is like a dictionary. It gives a unique ID (number) to every word.
# 'num_words=100' means we only care about the 100 most common words.
# 'oov_token' is what we use for words the model hasn't seen before.
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# This turns "Box now" into something like [4, 7]
sequences = tokenizer.texts_to_sequences(sentences)

# Neural Networks require every input to have the SAME length.
# 'maxlen=5' means our "sentence window" is exactly 5 words long.
# If a sentence is shorter, we add zeros at the end ('post' padding).
padded = pad_sequences(sequences, maxlen=5, padding='post')

# --- PART 3: Building the Brain (The Model Architecture) ---
# We build the model like a sandwich, layer by layer.
model = tf.keras.Sequential([
    # LAYER 1: Embedding
    # This is the most important part of NLP. It turns word IDs into vectors.
    # It learns that "fire" and "immediately" are "close" in meaning.
    tf.keras.layers.Embedding(input_dim=100, output_dim=16),
    
    # LAYER 2: Pooling
    # It collapses the 5-word sequence into a single "meaning" average.
    tf.keras.layers.GlobalAveragePooling1D(),
    
    # LAYER 3: Hidden Dense Layer
    # A layer of 8 "neurons" that looks for patterns in the meaning.
    tf.keras.layers.Dense(8, activation='relu'),
    
    # LAYER 4: Output Layer
    # One single neuron that gives a final score between 0 and 1.
    # 'sigmoid' is the math function that squashes any number into that 0-1 range.
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 'adam' is the optimizer (how the model adjusts its brain to get better).
# 'binary_crossentropy' is the loss function (how we measure how "wrong" the model is).
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- PART 4: Training (The Study Phase) ---
print("--- Training Deep Learning Model ---")
# 'epochs=10' means the model will read our 6 examples 10 times to learn.
model.fit(padded, labels, epochs=10, verbose=0)
print("Model trained successfully.")

# --- PART 5: Testing (Prediction) ---
# Let's try a sentence the model has NEVER seen.
test_radio = [input("Enter a message: ")]

# We MUST process the new sentence exactly like we did the training data.
test_seq = tokenizer.texts_to_sequences(test_radio)
test_padded = pad_sequences(test_seq, maxlen=5, padding='post')

# The model returns a probability (e.g., 0.85).
prediction = model.predict(test_padded)[0][0]

# If the score is higher than 0.5, we label it as Urgent.
status = "Urgent" if prediction > 0.5 else "Normal"

print("\n--- Deep Learning Result ---")
print("New Message: " + test_radio[0])
print("AI Prediction Score: " + str(round(prediction, 4)))
print("Final Status: " + status)