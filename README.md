# 🏎️ F1 Race Engineer AI (Production)

Welcome to the **F1 Race Engineer AI** project! 

This repository documents a complete journey from basic NLP concepts to a Production-Grade Deep Learning application, wrapped in verify "Cyber-Industrial" F1 Dashboard.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://f1-ai-dashboard.streamlit.app/)

**🔴 Live Demo:** [f1-ai-dashboard.streamlit.app](https://f1-ai-dashboard.streamlit.app/)

---

## 🚀 production Features (v2.0)
- **Synthetic Data Engine**: Generates 600+ unique training examples on the fly.
- **Deep Learning Core**: 3-Layer Neural Network (32-dim Embeddings) with validation monitoring.
- **Mission Control UI**: 
    - Full **F1 Telemetry Aesthetic** (Dark Mode, Orbitron Fonts, Neon Accents).
    - Real-time "Threat Probability" visualization.
    - Simulated "System Boot" training sequence.

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
    *Note: First run will download the 40MB spaCy model automatically.*

---

## 📚 Technical Stack
- **Frontend**: Streamlit (with custom CSS injection).
- **NLP**: spaCy (en_core_web_md).
- **ML**: TensorFlow/Keras (Sequential Model).
- **Ops**: Synthetic Data Generation for robust training.

**"Box, Box. We are ready for deployment."** 🏁
