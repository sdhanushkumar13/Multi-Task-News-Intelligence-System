📰 Multi-Task News Intelligence System

An end-to-end NLP system that performs Text Classification, Named Entity Recognition (NER), and Text Summarization on news articles using both from-scratch models and pretrained transformers, deployed on the cloud.

🚀 Project Overview

This project builds a unified NLP pipeline capable of:

📌 Text Classification – Categorizing news into domains like Politics, Business, Tech, Sports, etc.
🧠 Named Entity Recognition (NER) – Extracting entities such as Person, Organization, Location, and Date
✂️ Summarization – Generating concise summaries of long news articles

The system compares:
Traditional Machine Learning models
Custom Deep Learning models
Pretrained Transformer models (BERT, DistilBERT, T5, BART)

🧰 Tech Stack
Languages & Libraries: Python, Scikit-learn, PyTorch, TensorFlow, Hugging Face Transformers
NLP Techniques: BoW, TF-IDF, Word2Vec, Seq2Seq, BiLSTM-CRF
Frontend/UI: Streamlit
Cloud: AWS EC2, S3, RDS
Database: PostgreSQL
Tools: Git, SQLAlchemy

📂 Dataset
Microsoft PENS (Personalized News Headlines Dataset)

⚙️ Features
1. Text Classification
ML Models: Logistic Regression, SVM
DL Models: CNN, LSTM, BiLSTM
Transformers: BERT, DistilBERT
2. Named Entity Recognition (NER)
Rule-based baseline
BiLSTM / BiLSTM-CRF
Transformer-based NER
3. Summarization
Extractive (TF-IDF, TextRank)
Seq2Seq (LSTM with attention)
Transformers (T5, BART)

🔍 Workflow
Data Preprocessing
Text cleaning (remove HTML, URLs, special chars)
Tokenization & stopword removal
Feature engineering (BoW, TF-IDF, embeddings)
Model Building
ML, DL, and Transformer models for each task
Evaluation
Classification: Accuracy, Precision, Recall, F1
NER: F1-score (per entity type)
Summarization: ROUGE metrics
Application Development
Input: Text or file upload
Task selection: Classification / NER / Summarization
Model selection: ML / DL / Transformer
Output: Predictions, entities, summaries
Deployment
EC2 for hosting
S3 for model storage
RDS for logging user activity

🌐 Deployment Architecture
EC2 → Hosts the application
S3 → Stores trained models & artifacts
RDS → Stores logs and user interactions

📊 Evaluation Metrics
Classification: Accuracy, Precision, Recall, F1-score
NER: Precision, Recall, F1-score
Summarization: ROUGE-1, ROUGE-2, ROUGE-L

Source Code (Streamlit App)
Deployment Scripts (AWS / Hugging Face)

🧪 Results
Fully functional multi-task NLP web app
Cloud deployment with public access
Real-time predictions with logging capability
