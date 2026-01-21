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
    
    /* METRIC CARDS */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 5px;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #aaaaaa;
        font-size: 0.9rem;
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
        border-radius: 0px; /* Sharp edges for tech look */
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background: #ff2800;
        box-shadow: 0 0 20px #ff2800;
        color: black;
        border-color: #ff2800;
    }
    
    /* TEXT INPUTS */
    .stTextInput>div>div>input {
        background-color: rgba(0, 0, 0, 0.5);
        color: #00ffcc;
        border: 1px solid #333;
        font-family: 'Consolas', monospace;
    }
    .stTextArea>div>div>textarea {
        background-color: rgba(0, 0, 0, 0.5);
        color: #00ffcc;
        border: 1px solid #333;
        font-family: 'Consolas', monospace;
    }

    /* ALERTS & STATUS */
    .urgent-box {
        background: rgba(255, 0, 0, 0.1);
        border: 2px solid #ff0000;
        color: #ff4444;
        padding: 20px;
        text-align: center;
        border-radius: 5px;
        box-shadow: 0 0 20px rgba(255, 0, 0, 0.3);
        animation: pulse 1.5s infinite;
    }
    .normal-box {
        background: rgba(0, 255, 0, 0.1);
        border: 2px solid #00ff00;
        color: #00ffcc;
        padding: 20px;
        text-align: center;
        border-radius: 5px;
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.2);
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
    
    /* PROGRESS BAR */
    .stProgress > div > div > div > div {
        background-color: #ff2800;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER & TELEMETRY ---
c1, c2, c3 = st.columns([1, 4, 1])
with c1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1200px-F1.svg.png", width=100) # Simple F1 Logo placeholder or nothing
with c2:
    st.markdown("<h1 style='text-align: center;'>F1 RACE ENGINEER AI</h1>", unsafe_allow_html=True)

st.divider()

# Telemetry strip
col1, col2, col3, col4 = st.columns(4)
col1.metric("TRACK TEMP", "32.4°C", "1.2°C")
col2.metric("WIND SPEED", "4.2 km/h", "-0.5")
col3.metric("HUMIDITY", "45%", "Stable")
col4.metric("AI CONFIDENCE", "98.7%", "+0.2%")

st.divider()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🛠️ MISSION CONTROL")
    selection = st.radio(
        "SELECT MODULE:", 
        ["Syntax Analysis", "Entity Recognition", "Vector Embeddings", "Urgency Classifier (DL)"]
    )
    st.markdown("---")
    st.info("System Version: v2.0.4\nKernel: Neural-Link")

# --- LOADER ---
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        from spacy.cli import download
        download("en_core_web_md")
        return spacy.load("en_core_web_md")

nlp = load_nlp_model()

# --- SYNTHETIC DATA GENERATOR ---
def generate_f1_data(num_samples=600):
    urgent_subjects = ["Engine", "Brakes", "Tires", "Gearbox", "Clutch", "Hydraulics", "Battery", "Turbo", "Wing", "Suspension"]
    urgent_verbs = ["failing", "on fire", "vibrating", "exploding", "smoking", "overheating", "broken", "leaking", "stuck", "losing power"]
    urgent_contexts = ["badly", "critical", "box now", "immediately", "terminal", "failure", "danger", "stop stop", "red alarm", "abort"]

    normal_subjects = ["Gap", "Pace", "Wind", "Weather", "Tire temp", "Fuel", "Strategy", "Radio", "Telemetry", "Sector 1"]
    normal_verbs = ["is", "looks", "seems", "stays", "remains", "feeling", "holding", "reporting", "reading", "confirmed"]
    normal_contexts = ["stable", "good", "okay", "consistent", "green", "normal", "acceptable", "fine", "nominal", "according to plan"]

    sentences = []
    labels = []

    for _ in range(num_samples // 2):
        s = f"{random.choice(urgent_subjects)} {random.choice(urgent_verbs)} {random.choice(urgent_contexts)}"
        sentences.append(s)
        labels.append(1)
        
        s = f"{random.choice(normal_subjects)} {random.choice(normal_verbs)} {random.choice(normal_contexts)}"
        sentences.append(s)
        labels.append(0)
    
    combined = list(zip(sentences, labels))
    random.shuffle(combined)
    return zip(*combined)

# --- SECTION 1: SYNTAX ANALYSIS ---
if selection == "Syntax Analysis":
    st.markdown("### 1. TELEMETRY SYNTAX PARSING")
    st.caption("Deconstruct driver messages into grammatical dependencies.")
    
    text_input = st.text_area("INCOMING TRANSMISSION:", "Engine is overheating badly", height=80)
    
    if st.button("RUN DIAGNOSTICS"):
        doc = nlp(text_input)
        html = displacy.render(doc, style="dep", page=False, options={"bg": "#1e1e1e", "color": "#00ffcc"})
        st.write(html, unsafe_allow_html=True)

# --- SECTION 2: NER ---
elif selection == "Entity Recognition":
    st.markdown("### 2. ENTITY EXTRACTION PROTOCOL")
    st.caption("Identify critical components and actions in voice comms.")

    @st.cache_resource
    def setup_ner(_nlp):
        if "entity_ruler" not in _nlp.pipe_names:
            ruler = _nlp.add_pipe("entity_ruler", before="ner")
            patterns = [
                {"label": "PART", "pattern": "Engine"}, {"label": "PART", "pattern": "Tires"},
                {"label": "STATUS", "pattern": "Critical"}, {"label": "STATUS", "pattern": "Stable"},
                {"label": "ACTION", "pattern": "Box"}, {"label": "ACTION", "pattern": "Push"}
            ]
            ruler.add_patterns(patterns)
        return _nlp

    nlp = setup_ner(nlp)
    text_input = st.text_area("INCOMING TRANSMISSION:", "Box now for new Tires", height=80)
    
    if st.button("SCAN ENTITIES"):
        doc = nlp(text_input)
        # Custom coloring for entities
        options = {"colors": {"PART": "#ff2800", "STATUS": "#ffff00", "ACTION": "#00ffcc"}}
        html = displacy.render(doc, style="ent", page=False, options=options)
        st.write(html, unsafe_allow_html=True)

# --- SECTION 3: EMBEDDINGS ---
elif selection == "Vector Embeddings":
    st.markdown("### 3. SEMANTIC VECTOR ANALYSIS")
    st.caption("Compare terminology similarity in high-dimensional space.")
    
    c1, c2 = st.columns(2)
    w1 = c1.text_input("VECTOR A", "Tires")
    w2 = c2.text_input("VECTOR B", "Vibration")
    
    if st.button("CALCULATE COSINE SIMILARITY"):
        score = nlp(w1).similarity(nlp(w2))
        st.metric("SIMILARITY SCORE", f"{score:.4f}")
        st.progress(score)

# --- SECTION 4: DEEP LEARNING (PRO) ---
elif selection == "Urgency Classifier (DL)":
    st.markdown("### 4. PREDICTIVE URGENCY MODEL")
    st.caption("Deep Learning classification of driver distress signals.")
    
    # Check if model exists logic could provide persistence, but we train fresh for demo
    
    @st.cache_resource
    def train_pro_model():
        sentences, labels = generate_f1_data(800)
        sentences = list(sentences)
        labels = np.array(list(labels))

        tokenizer = Tokenizer(num_words=600, oov_token="<OOV>")
        tokenizer.fit_on_texts(sentences)
        sequences = tokenizer.texts_to_sequences(sentences)
        padded = pad_sequences(sequences, maxlen=10, padding='post')

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=600, output_dim=32),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Simulate training steps for UI effect
        return model, tokenizer, padded, labels

    if 'model' not in st.session_state:
        if st.button("INITIALIZE NEURAL NETWORK"):
            with st.status("BOOTING AI SUBSYSTEMS...", expanded=True) as status:
                st.write("Generating Synthetic Telemetry Data...")
                time.sleep(1)
                model_base, tokenizer_base, padded, labels = train_pro_model()
                
                st.write("Vectorizing Text Inputs...")
                time.sleep(0.5)
                
                st.write("Training Epochs (Adam Optimizer)...")
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Actual Training
                model_base.fit(padded, labels, epochs=20, verbose=0)
                
                st.session_state['model'] = model_base
                st.session_state['tokenizer'] = tokenizer_base
                status.update(label="SYSTEM ONLINE. MODEL READY.", state="complete", expanded=False)
                st.rerun()

    else:
        st.success("MODEL ONLINE - READY FOR CLASSIFICATION")
        
        user_msg = st.text_input("DRIVER VOICE INPUT:", "Brakes are failing critical")
        
        if st.button("ANALYZE PRIORITY"):
            model = st.session_state['model']
            tokenizer = st.session_state['tokenizer']
            
            seq = tokenizer.texts_to_sequences([user_msg])
            pad = pad_sequences(seq, maxlen=10, padding='post')
            pred = model.predict(pad)[0][0]
            
            st.divider()
            
            c_res, c_prob = st.columns([2, 1])
            
            with c_res:
                if pred > 0.8:
                    st.markdown(f'''
                        <div class="urgent-box">
                            <h1>⚠️ URGENT STOP REVIEW</h1>
                            <p>CRITICAL FAILURE DETECTED</p>
                        </div>
                    ''', unsafe_allow_html=True)
                elif pred < 0.5:
                    st.markdown(f'''
                        <div class="normal-box">
                            <h1>✅ SYSTEM NOMINAL</h1>
                            <p>NO ACTION REQUIRED</p>
                        </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ CAUTION: TELEMETRY UNCLEAR")
            
            with c_prob:
                st.metric("THREAT PROBABILITY", f"{pred:.2%}")
                st.progress(float(pred))
