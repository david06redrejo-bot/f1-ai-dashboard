import streamlit as st
import spacy
from spacy import displacy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import random
import time

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="F1 Race Engineer AI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (F1 AESTHETIC) ---
st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');

    /* GENERAL SETTINGS */
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 50%, #1a1a1a 0%, #000000 100%);
        color: #e0e0e0;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* MASK STREAMLIT ELEMENTS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* TYPOGRAPHY */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    /* ACTIONS & METRICS */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 5px;
        backdrop-filter: blur(5px);
    }
    div[data-testid="stMetricValue"] {
        color: #00ffcc; /* Cyan for data */
        font-family: 'Orbitron', monospace;
    }

    /* BUTTONS */
    .stButton>button {
        color: white;
        background: transparent;
        border: 2px solid #ff2800; /* Ferrari Red */
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        width: 100%;
        border-radius: 0px;
    }
    .stButton>button:hover {
        background: #ff2800;
        color: black;
        box-shadow: 0 0 15px #ff2800;
    }
    
    /* INPUTS */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: rgba(0, 0, 0, 0.5);
        color: #00ffcc;
        border: 1px solid #333;
        font-family: 'Consolas', monospace;
    }

    /* ALERTS */
    .urgent-box {
        background: rgba(255, 0, 0, 0.15);
        border: 2px solid #ff0000;
        color: #ff4444;
        padding: 20px;
        text-align: center;
        border-radius: 5px;
        box-shadow: 0 0 20px rgba(255, 0, 0, 0.4);
        margin-top: 20px;
        animation: pulse 2s infinite;
    }
    .normal-box {
        background: rgba(0, 255, 0, 0.15);
        border: 2px solid #00ff00;
        color: #00ffcc;
        padding: 20px;
        text-align: center;
        border-radius: 5px;
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
        margin-top: 20px;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
c1, c2 = st.columns([1, 6])
with c1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1200px-F1.svg.png", width=80)
with c2:
    st.markdown("<h1 style='padding-top: 10px;'>F1 RACE ENGINEER AI <span style='font-size: 0.5em; color: #ff2800;'>PRO v2.1</span></h1>", unsafe_allow_html=True)

st.divider()

# --- TELEMETRY DECK (RESTORED) ---
cols = st.columns(4)
cols[0].metric("TRACK TEMP", "32.4°C", "1.2°C")
cols[1].metric("WIND SPEED", "4.2 km/h", "-0.5")
cols[2].metric("HUMIDITY", "45%", "Stable")
cols[3].metric("AI CONFIDENCE", "99.1%", "+0.4%")

st.divider()

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.markdown("### 🛠️ MISSION CONTROL")
    selection = st.radio(
        "ACTIVE MODULE:", 
        ["Syntax Analysis", "Entity Recognition", "Vector Embeddings", "Urgency Classifier (DL)"]
    )
    st.markdown("---")
    st.caption("SYSTEM STATUS: ONLINE")

# --- SPACY LOADER ---
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        from spacy.cli import download
        download("en_core_web_md")
        return spacy.load("en_core_web_md")

with st.spinner("INITIALIZING AI SUBSYSTEMS..."):
    nlp = load_nlp_model()

# --- SYNTHETIC DATA GENERATOR ---
def generate_f1_data(num_samples=600):
    # Urgent Components
    urgent_subjects = ["Engine", "Brakes", "Tires", "Gearbox", "Clutch", "Hydraulics", "Battery", "Turbo", "Rear Wing", "Suspension", "MGU-K"]
    urgent_verbs = ["failing", "on fire", "vibrating", "exploding", "smoking", "overheating", "broken", "leaking", "stuck", "losing power", "gone"]
    urgent_contexts = ["badly", "critical", "box now", "immediately", "terminal", "failure", "danger", "stop stop", "red alarm", "abort", "emergency"]

    # Normal Components
    normal_subjects = ["Gap", "Pace", "Wind", "Weather", "Tire temp", "Fuel", "Strategy", "Radio", "Telemetry", "Sector 1", "Balance"]
    normal_verbs = ["is", "looks", "seems", "stays", "remains", "feeling", "holding", "reporting", "reading", "confirmed", "checking"]
    normal_contexts = ["stable", "good", "okay", "consistent", "green", "normal", "acceptable", "fine", "nominal", "according to plan", "steady"]

    sentences = []
    labels = []

    # Generate Balanced Dataset
    for _ in range(num_samples // 2):
        # Case 1: Urgent (Label 1)
        s_urgent = f"{random.choice(urgent_subjects)} {random.choice(urgent_verbs)} {random.choice(urgent_contexts)}"
        sentences.append(s_urgent)
        labels.append(1)
        
        # Case 2: Normal (Label 0)
        s_normal = f"{random.choice(normal_subjects)} {random.choice(normal_verbs)} {random.choice(normal_contexts)}"
        sentences.append(s_normal)
        labels.append(0)
    
    # Shuffle Data
    combined = list(zip(sentences, labels))
    random.shuffle(combined)
    return zip(*combined)

# --- DEEP LEARNING MODEL TRAINER ---
@st.cache_resource
def train_production_model():
    """
    Trains the Keras model once and caches it.
    """
    # 1. Generate Synthetic Data
    X_text, y_labels = generate_f1_data(600)
    X_text = list(X_text)
    y_labels = np.array(list(y_labels))
    
    # 2. Tokenization
    # Increased vocab size to 1000 to capture all variations
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_text)
    
    sequences = tokenizer.texts_to_sequences(X_text)
    # Padded length matches sentence structure (Subj + Verb + Context = ~3-5 words, maxlen=10 is safe)
    padded_sequences = pad_sequences(sequences, maxlen=10, padding='post')
    
    # 3. Model Architecture
    model = tf.keras.Sequential([
        # Embedding: 1000 vocab, 32 dims (Better feature representation)
        tf.keras.layers.Embedding(input_dim=1000, output_dim=32),
        
        # GlobalAveragePooling: Efficiently flattening for short text
        tf.keras.layers.GlobalAveragePooling1D(),
        
        # Hidden Layer: 16 units with ReLU
        tf.keras.layers.Dense(16, activation='relu'),
        
        # Output: Sigmoid for Binary Classification (0=Normal, 1=Urgent)
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 4. Training
    # 30 Epochs ensure convergence on this small-ish dataset
    model.fit(padded_sequences, y_labels, epochs=30, verbose=0)
    
    return model, tokenizer

# --- MAIN APP LOGIC ---

if selection == "Syntax Analysis":
    st.markdown("### 1. SYNTAX TELEMETRY")
    st.caption("Parses grammatical structure of incoming transmissions.")
    
    text_input = st.text_area("INCOMING MSG:", "Engine is overheating badly", height=80)
    
    if st.button("RUN PARSER"):
        doc = nlp(text_input)
        # Custom visualizer options
        options = {"bg": "#0e1117", "color": "#00ffcc", "font": "Rajdhani"}
        html = displacy.render(doc, style="dep", page=False, options=options)
        st.write(html, unsafe_allow_html=True)
        
        # Extract Subject/Action
        st.markdown("#### KEY ENTITIES")
        cols = st.columns(2)
        root = [token.text.upper() for token in doc if token.dep_ == "ROOT"]
        subj = [token.text.upper() for token in doc if token.dep_ == "nsubj"]
        cols[0].metric("ACTION (ROOT)", root[0] if root else "N/A")
        cols[1].metric("SUBJECT", subj[0] if subj else "N/A")

elif selection == "Entity Recognition":
    st.markdown("### 2. ENTITY RECOGNITION")
    st.caption("Identifies critical F1 terminology.")
    
    # Setup Entity Ruler
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "COMPONENT", "pattern": "Engine"}, {"label": "COMPONENT", "pattern": "Tires"},
            {"label": "COMPONENT", "pattern": "Brakes"}, {"label": "COMPONENT", "pattern": "Gearbox"},
            {"label": "STRATEGY", "pattern": "Box"}, {"label": "STRATEGY", "pattern": "Pit"},
            {"label": "STATUS", "pattern": "Critical"}, {"label": "STATUS", "pattern": "Failing"}
        ]
        ruler.add_patterns(patterns)
        
    text_input = st.text_area("INCOMING MSG:", "Box now, Brakes are failing", height=80)
    
    if st.button("SCAN MESSAGE"):
        doc = nlp(text_input)
        colors = {"COMPONENT": "#ff2800", "STRATEGY": "#00ffcc", "STATUS": "#ffff00"}
        options = {"colors": colors}
        html = displacy.render(doc, style="ent", page=False, options=options)
        st.write(html, unsafe_allow_html=True)

elif selection == "Vector Embeddings":
    st.markdown("### 3. SEMANTIC VECTORS")
    st.caption("Analyzes conceptual similarity.")
    
    c1, c2 = st.columns(2)
    w1 = c1.text_input("TERM A", "Tires")
    w2 = c2.text_input("TERM B", "Vibration")
    
    if st.button("COMPARE"):
        score = nlp(w1).similarity(nlp(w2))
        st.metric("Similarity Coefficient", f"{score:.4f}")
        st.progress(float(score))

elif selection == "Urgency Classifier (DL)":
    st.markdown("### 4. URGENCY CLASSIFIER (NEURAL NET)")
    st.caption("Predicts driver distress levels using Deep Learning.")
    
    # Load Model (Cached)
    with st.spinner("Initializing Neural Core..."):
        model, tokenizer = train_production_model()
    
    st.success("NEURAL CORE ONLINE")
    
    user_msg = st.text_input("DRIVER VOICE INPUT:", "Engine is on fire")
    
    if st.button("PREDICT PRIORITY"):
        # Preprocess
        seq = tokenizer.texts_to_sequences([user_msg])
        pad = pad_sequences(seq, maxlen=10, padding='post')
        
        # Predict
        prediction = model.predict(pad)[0][0]
        
        # Render Result
        st.divider()
        c1, c2 = st.columns([2, 1])
        
        with c1:
            if prediction > 0.8:
                st.markdown(f"""
                <div class="urgent-box">
                    <h1>⚠️ CRITICAL ALERT</h1>
                    <h3>{user_msg.upper()}</h3>
                    <p>IMMEDIATE ACTION REQUIRED</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction < 0.5:
                st.markdown(f"""
                <div class="normal-box">
                    <h1>✅ SYSTEM NOMINAL</h1>
                    <h3>{user_msg.upper()}</h3>
                    <p>NO ISSUE DETECTED</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                 st.warning("⚠️ UNCLEAR TRANSMISSION - MANUAL REVIEW")
        
        with c2:
            st.metric("URGENCY PROBABILITY", f"{prediction:.2%}")
            st.progress(float(prediction))
