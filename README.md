# 🏎️ F1 Race Engineer AI (Production v2.2)

Welcome to the **F1 Race Engineer AI** project! 

This repository documents a complete journey from basic NLP concepts to a Production-Grade Deep Learning application, wrapped in verify "Cyber-Industrial" F1 Dashboard.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://f1-ai-dashboard.streamlit.app/)

**🔴 Live Demo:** [f1-ai-dashboard.streamlit.app](https://f1-ai-dashboard.streamlit.app/)

---

## 🚀 Upgrade to v2.2 (System Audit)
Following a comprehensive deep-dive, the system has been upgraded:
- **Bias Correction**: Mitigated "Mode Collapse" by balancing the training set with short routine phrases ("Ok copy") and implementing `mask_zero` masking layers.
- **Enhanced NER**: Entity Recognition is now **case-insensitive** and detects Drivers (Hamilton, Verstappen, etc.).
- **Production UI**: Dark Mode "Mission Control" interface with Orbitron typography and glassmorphism.
- **Robustness**: Auto-detects and installs NLP models (`en_core_web_md`) and handles missing dependencies gracefully.

---

## 📂 Project Structure

- **`app.py`**: The Main Dashboard (Streamlit).
- **`project_days/`**: The code evolution history.
    - `day1_syntax.py`: Syntax Trees.
    - `day2_ner_lm.py.py`: Entity Recognition.
    - `day3_embeddings.py`: Semantic Vectors.
    - `day4_deep_learning.py`: Basic Keras implementation.
- **`project_visualization/`**: Standalone scripts that generate clear HTML explanations of the algorithms.
    - `day2_visualize.py`: NER & Bigram Visualizer.
    - `day3_visualize.py`: Word Embedding Heatmaps.
    - `day4_visualize.py`: Deep Learning Architecture Diagram.

---

## 🛠️ How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Launch Dashboard**:
    ```bash
    streamlit run app.py
    ```
    *Note: First run will download the key AI models automatically.*

---

## 📚 Technical Stack
- **Frontend**: Streamlit (Custom CSS, Orbital Fonts).
- **NLP**: spaCy (en_core_web_md).
- **ML**: TensorFlow/Keras (3-Layer Sequential Network).
- **Ops**: Synthetic Data Generation for robust training.

**"Box, Box. We are ready for deployment."** 🏁
