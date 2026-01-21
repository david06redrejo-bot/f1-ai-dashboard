# 🏎️ F1 Race Engineer AI (Production)

Welcome to the **F1 Race Engineer AI** project! 

This repository documents a complete journey from basic NLP concepts to a Production-Grade Deep Learning application, wrapped in verify "Cyber-Industrial" F1 Dashboard.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://f1-ai-dashboard.streamlit.app/)

**🔴 Live Demo:** [f1-ai-dashboard.streamlit.app](https://f1-ai-dashboard.streamlit.app/)

---

## 🚀 Upgrade to v2.1 (Technical Audit)
Following a comprehensive technical audit, the system has been upgraded to **Production Grade**:
- **Fixed Model Convergence**: Now trains on **600+ synthetic data points** (up from 10) to ensure high accuracy.
- **Robustness**: Auto-detects and installs NLP models (`en_core_web_md`) on cloud environments.
- **Efficiency**: Implemented smart caching (`@st.cache_resource`) for the Neural Network training pipeline.
- **UI/UX**: Deep "Cyber-Industrial" aesthetic with real-time threat visualization.

---

## 📂 Project Structure

- **`app.py`**: The Main Dashboard (Streamlit).
- **`project_days/`**: The code evolution history.
    - `day1_syntax.py`: Syntax Trees.
    - `day2_ner_lm.py.py`: Entity Recognition.
    - `day3_embeddings.py`: Semantic Vectors.
    - `day4_deep_learning.py`: Basic Keras implementation.

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
