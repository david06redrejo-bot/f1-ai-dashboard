import os
from pathlib import Path
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- PART 1: Model Setup & Training (Based on day4_deep_learning.py) ---

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    HAS_TF = True
    
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

    labels = np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])

    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=5, padding='post')

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=100, output_dim=16),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(padded, labels, epochs=10, verbose=0)
    
except ModuleNotFoundError:
    print("Warning: TensorFlow not found. Using mock data for visualization.")
    HAS_TF = False
    
# --- PART 2: Visualization Generation ---

# A. Visualizing a Prediction
test_msg = "Engine failure box now"

if HAS_TF:
    test_seq = tokenizer.texts_to_sequences([test_msg])
    test_padded = pad_sequences(test_seq, maxlen=5, padding='post')
    prediction = model.predict(test_padded)[0][0]
    token_str = str(test_padded[0])
else:
    # Mock data
    prediction = 0.88
    token_str = "[Mock Tokens: 4, 12, 5, 0, 0]"

status = "Urgent" if prediction > 0.5 else "Normal"
prob_percent = round(prediction * 100, 2)


# B. Architecture Diagram (SVG)


# B. Architecture Diagram (SVG)
# Simple representation of flow
svg_content = """
<svg width="600" height="200" xmlns="http://www.w3.org/2000/svg">
  <!-- Input -->
  <rect x="10" y="80" width="80" height="40" fill="#add8e6" stroke="black" />
  <text x="50" y="105" text-anchor="middle">Input</text>
  
  <!-- Arrow -->
  <line x1="90" y1="100" x2="130" y2="100" stroke="black" marker-end="url(#arrow)" />
  
  <!-- Embedding -->
  <rect x="130" y="60" width="100" height="80" fill="#90ee90" stroke="black" />
  <text x="180" y="105" text-anchor="middle">Embedding</text>
  
  <!-- Arrow -->
  <line x1="230" y1="100" x2="270" y2="100" stroke="black" marker-end="url(#arrow)" />
  
  <!-- Pooling -->
  <rect x="270" y="70" width="80" height="60" fill="#ffa07a" stroke="black" />
  <text x="310" y="105" text-anchor="middle">Pool</text>
  
  <!-- Arrow -->
  <line x1="350" y1="100" x2="390" y2="100" stroke="black" marker-end="url(#arrow)" />

  <!-- Dense -->
  <rect x="390" y="80" width="60" height="40" fill="#dda0dd" stroke="black" />
  <text x="420" y="105" text-anchor="middle">Dense</text>
  
  <!-- Arrow -->
  <line x1="450" y1="100" x2="490" y2="100" stroke="black" marker-end="url(#arrow)" />

  <!-- Output -->
  <circle cx="520" cy="100" r="20" fill="#ffcccb" stroke="black" />
  <text x="520" y="105" text-anchor="middle">Out</text>
  
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="0" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#000" />
    </marker>
  </defs>
</svg>
"""

# HTML Content
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Day 4: Deep Learning Visualization</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; }}
        .container {{ display: flex; flex-direction: column; align-items: center; }}
        .prediction-box {{
            border: 2px solid #333;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            width: 50%;
            text-align: center;
            background-color: #f9f9f9;
        }}
        .score-bar-bg {{
            width: 100%;
            height: 30px;
            background-color: #ddd;
            border-radius: 15px;
            margin-top: 10px;
            overflow: hidden;
        }}
        .score-bar-fill {{
            height: 100%;
            width: {prob_percent}%;
            background-color: {'#ff4444' if prediction > 0.5 else '#44cc44'};
            text-align: right;
            line-height: 30px;
            color: white;
            padding-right: 10px;
            transition: width 1s ease-in-out;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Day 4: Deep Learning Architecture</h1>
        <p>A sequential model that translates words to math vectors and predicts urgency.</p>
        
        {svg_content}
        
        <div class="prediction-box">
            <h2>Live Prediction Test</h2>
            <p><strong>Input Message:</strong> "{test_msg}"</p>
            <p><strong>Processed Tokens:</strong> {token_str}</p>
            
            <h3>Prediction Result: {status}</h3>
            <div class="score-bar-bg">
                <div class="score-bar-fill">{prob_percent}%</div>
            </div>
            <p>Probability of Urgency</p>
        </div>
    </div>
</body>
</html>
"""

output_file = Path("day4_deep_learning.html")
output_file.write_text(html_content, encoding="utf-8")

print(f"Successfully created visualization at: {output_file.absolute()}")
