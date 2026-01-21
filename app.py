import streamlit as st
import spacy
from spacy import displacy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from collections import Counter, defaultdict

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="F1 Race Engineer AI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Cyber-Industrial" look (Dark Mode friendly)
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #ff4b4b; /* F1 Red */
    }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏎️ F1 Race Engineer AI Dashboard")

# --- SIDEBAR ---
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Syntax Analysis", "Entity Recognition", "Vector Embeddings", "Urgency Classifier (DL)"])

# --- LOADER ---
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        st.error("Model 'en_core_web_md' not found. Please run: python -m spacy download en_core_web_md")
        return None

nlp = load_nlp_model()

if not nlp:
    st.stop()

# --- SECTION 1: SYNTAX ANALYSIS ---
if selection == "Syntax Analysis":
    st.header("1. Syntax Analysis (Dependency Trees)")
    st.markdown("Visualize the grammatical structure of radio messages.")
    
    text_input = st.text_area("Enter Radio Message:", "Hamilton is complaining about the tires", height=100)
    
    if st.button("Analyze Syntax"):
        doc = nlp(text_input)
        
        # Display Displacy Tree
        html = displacy.render(doc, style="dep", page=False)
        st.write(html, unsafe_allow_html=True)
        
        # Highlight Key Parts
        st.subheader("Key Components")
        root = [token.text for token in doc if token.dep_ == "ROOT"]
        nsubj = [token.text for token in doc if token.dep_ == "nsubj"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Action (ROOT):** {', '.join(root)}")
        with col2:
            st.info(f"**Subject (nsubj):** {', '.join(nsubj)}")

# --- SECTION 2: NER & N-GRAMS ---
elif selection == "Entity Recognition":
    st.header("2. Entity Recognition & Autocomplete")
    
    # Custom Entity Ruler Setup
    @st.cache_resource
    def setup_ner(_nlp):
        if "entity_ruler" not in _nlp.pipe_names:
            ruler = _nlp.add_pipe("entity_ruler", before="ner")
            patterns = [
                {"label": "DRIVER", "pattern": "Hamilton"},
                {"label": "DRIVER", "pattern": "Verstappen"},
                {"label": "DRIVER", "pattern": "Leclerc"},
                {"label": "DRIVER", "pattern": "Norris"},
                {"label": "COMPOUND", "pattern": "Softs"},
                {"label": "COMPOUND", "pattern": "Hards"},
                {"label": "COMPOUND", "pattern": "Inters"},
                {"label": "COMPOUND", "pattern": "Wets"},
                {"label": "ACTION", "pattern": "Box"},
                {"label": "ACTION", "pattern": "Pit"}
            ]
            ruler.add_patterns(patterns)
        return _nlp

    nlp = setup_ner(nlp)

    text_input = st.text_area("Enter Text for NER:", "Hamilton needs to Box for Softs", height=100)
    
    if st.button("Extract Entities"):
        doc = nlp(text_input)
        # Verify if visualization works (ent style)
        html = displacy.render(doc, style="ent", page=False)
        st.write(html, unsafe_allow_html=True)
        
        with st.expander("Raw Entity Data"):
            for ent in doc.ents:
                st.write(f"**{ent.text}** -> {ent.label_}")

    st.markdown("---")
    st.subheader("Simple N-Gram Prediction")
    
    corpus_text = st.text_area("Training Corpus:", "Box box for softs. Box now for hards. Box for new tires. Hamilton is pitting now.", height=70)
    
    # Train Bigram
    model = defaultdict(Counter)
    tokens = [t.text.lower() for t in nlp(corpus_text) if not t.is_punct]
    for i in range(len(tokens) - 1):
        model[tokens[i]][tokens[i+1]] += 1
        
    word_input = st.text_input("Predict next word after:", "Box")
    
    if st.button("Predict"):
        word_lower = word_input.lower()
        if word_lower in model:
            next_word = model[word_lower].most_common(1)[0][0]
            st.success(f"Prediction: **{next_word}**")
        else:
            st.warning("Word not found in corpus.")

# --- SECTION 3: EMBEDDINGS ---
elif selection == "Vector Embeddings":
    st.header("3. Semantic Similarity (Embeddings)")
    
    col1, col2 = st.columns(2)
    with col1:
        w1 = st.text_input("Word/Phrase 1", "Tires")
    with col2:
        w2 = st.text_input("Word/Phrase 2", "Rubber")
        
    if st.button("Calculate Similarity"):
        doc1 = nlp(w1)
        doc2 = nlp(w2)
        score = doc1.similarity(doc2)
        
        st.metric(label="Cosine Similarity", value=f"{score:.4f}")
        
        if score > 0.7:
            st.success("High Similarity! Using similar context.")
        elif score > 0.4:
            st.warning("Moderate Similarity.")
        else:
            st.error("Low Similarity. Different contexts.")

# --- SECTION 4: DEEP LEARNING ---
elif selection == "Urgency Classifier (DL)":
    st.header("4. Deep Learning Classifier")
    st.markdown("Classify radio messages as **Urgent** or **Routine**.")

    # Cache the model training so it doesn't run on every interaction
    @st.cache_resource
    def train_dl_model():
        sentences = [
            "Box now, engine failing", "Stop the car immediately", "Fire at the back", 
            "I have a puncture", "Brakes are not working", "Crash in sector two",
            "The tires are slightly warm", "Keep pushing for two laps", "We are discussing the strategy",
            "Gap to leader is 5 seconds", "Battery charge is high", "DRS will be enabled"
        ]
        labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        
        # Tokenizer
        tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
        tokenizer.fit_on_texts(sentences)
        sequences = tokenizer.texts_to_sequences(sentences)
        padded = pad_sequences(sequences, maxlen=5, padding='post')
        
        # Model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=100, output_dim=16),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(padded, labels, epochs=20, verbose=0)
        
        return model, tokenizer

    model, tokenizer = train_dl_model()
    st.success("Model Trained & Loaded!")
    
    dl_input = st.text_input("Enter Message to Classify:", "Engine is on fire")
    
    if st.button("Classify Urgency"):
        seq = tokenizer.texts_to_sequences([dl_input])
        padded = pad_sequences(seq, maxlen=5, padding='post')
        
        pred = model.predict(padded)[0][0]
        
        st.write(f"Raw Prediction Score: {pred:.4f}")
        
        if pred > 0.5:
            st.error(f"⚠️ URGENT (Confidence: {pred:.2%})")
        else:
            st.success(f"✅ ROUTINE (Confidence: {(1-pred):.2%})")

