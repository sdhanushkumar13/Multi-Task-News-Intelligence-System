import streamlit as st
import joblib
import pickle
import numpy as np
import os
import re
import string
import nltk
import torch
import json
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
from TorchCRF import CRF
from transformers import BertTokenizerFast, BertForSequenceClassification, BertForTokenClassification
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, AdditiveAttention, Concatenate
from tensorflow.keras.models import Model
import psycopg2
import boto3

os.environ["TMPDIR"] = "/home/ec2-user/tmp"
os.environ["TEMP"] = "/home/ec2-user/tmp"
os.environ["TMP"] = "/home/ec2-user/tmp"

#--------------------------------
# RDS SETUP
#--------------------------------
DB_HOST = "nlp-db.cx6w4cqw67ox.ap-south-1.rds.amazonaws.com"
DB_NAME = "nlp_logs"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")

if DB_PASSWORD is None:
    st.error(" DB_PASSWORD not set in environment")
    st.stop()

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=5432
    )

def log_to_db(user_id, task, model_family, model_name, input_length, output, error):
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO inference_logs (
                user_id, task_type, model_family, model_name, input_length, output, error_flag
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (user_id, task, model_family, model_name, input_length, output, error))

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print("DB ERROR:", e)

#--------------------------------
# S3 Setup
# -------------------------------
s3 = boto3.client(
    "s3",
    config=boto3.session.Config(
        retries={"max_attempts": 5},
        connect_timeout=5,
        read_timeout=60
    )
)

BUCKET_NAME = "nlp-multitask-models-dk23"

LOCAL_MODEL_DIR = "/home/ec2-user/models"

def download_all_models():
    
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="models/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.endswith("/") or key.endswith("eval_data.pkl"):
                continue

            local_path = os.path.join(
                LOCAL_MODEL_DIR,
                key.replace("models/", "")
            )
            print(f"Downloading: {key}")

            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            s3.download_file(BUCKET_NAME, key, local_path)

    print("✅ All models downloaded from S3")

BASE_DIR = LOCAL_MODEL_DIR

print(" BASE_DIR =", BASE_DIR)

@st.cache_resource
def setup_models():
    download_all_models()
    return True

setup_models()

# -------------------------------
# CLASSIFICATION
# -------------------------------
ML_MODEL_PATH = os.path.join(BASE_DIR, "classification", "ml", "classify_ML_model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "classification", "le", "label_encoder.pkl")

DL_MODEL_PATH = os.path.join(BASE_DIR, "classification", "dl", "BiLSTM_GloVe.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "classification", "dl", "tokenizer.pkl")

PT_MODEL_PATH = os.path.join(BASE_DIR, "classification", "pt", "bert_model")


# -------------------------------
# NER
# -------------------------------
NER_MODEL_PATH = os.path.join(BASE_DIR, "ner", "dl", "best_model_crf.pt")
WORD2IDX_PATH = os.path.join(BASE_DIR, "ner", "dl", "word2idx.pkl")
IDX2TAG_PATH = os.path.join(BASE_DIR, "ner", "dl", "idx2tag.pkl")

NER_PT_MODEL_PATH = os.path.join(BASE_DIR, "ner", "pt", "bert_ner_model")
NER_PT_META_PATH = os.path.join(BASE_DIR, "ner", "pt", "id2tag_small.pkl")


# -------------------------------
# SUMMARIZATION
# -------------------------------
SUMM_MODEL_PATH = os.path.join(BASE_DIR, "summarization", "dl", "seq2seq_model.h5")
SUMM_ENCODER_PATH = os.path.join(BASE_DIR, "summarization", "dl", "encoder_model.keras")
SUMM_DECODER_PATH = os.path.join(BASE_DIR, "summarization", "dl", "decoder_model.keras")
SUMM_TOKENIZER_PATH = os.path.join(BASE_DIR, "summarization", "dl", "tokenizer.pkl")
SUMM_CONFIG_PATH = os.path.join(BASE_DIR, "summarization", "dl", "config.json")

SUMM_PT_PATH = os.path.join(BASE_DIR, "summarization", "pt", "BART-Large-CNN_model")

required_files = [
    ML_MODEL_PATH,
    LABEL_ENCODER_PATH,
    DL_MODEL_PATH,
    TOKENIZER_PATH,
    
    NER_MODEL_PATH,
    WORD2IDX_PATH,
    IDX2TAG_PATH,
    
    NER_PT_META_PATH,
    
    SUMM_ENCODER_PATH,
    SUMM_DECODER_PATH,
    SUMM_TOKENIZER_PATH,
    SUMM_CONFIG_PATH
]

required_dirs = [
    PT_MODEL_PATH,
    NER_PT_MODEL_PATH,
    SUMM_PT_PATH
]

# Check files
for path in required_files:
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()

# Check directories
for path in required_dirs:
    if not os.path.isdir(path):
        st.error(f"Missing model folder: {path}")
        st.stop()



# -------------------------------
# SETUP
# -------------------------------
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

NER_MAX_LEN = 60

MAX_LEN = 150
PT_MAX_LEN = 96


import re
import string

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    text = re.sub(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", "", text
    )

    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", " ", text)

    allowed = set(string.ascii_letters + string.digits + " .,!?")
    text = "".join(ch for ch in text if ch in allowed)

    text = text.lower()
    text = " ".join(text.split())

    return text

# -------------------------------
# DEFINE NER DL MODEL CLASS
# -------------------------------
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, input_ids):
        mask = (input_ids != 0)

        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        emissions = self.fc(x)

        return self.crf.decode(emissions, mask=mask)

# -------------------------------
# LOAD MODELS 
# -------------------------------

@st.cache_resource
def load_ml():
    model = joblib.load(ML_MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return model, le

@st.cache_resource
def load_dl():
    model = load_model(DL_MODEL_PATH)
    
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    #tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))
    return model, tokenizer

@st.cache_resource
def load_pt_classifier():

    tokenizer = BertTokenizerFast.from_pretrained(PT_MODEL_PATH)

    model = BertForSequenceClassification.from_pretrained(PT_MODEL_PATH)

    model.to(device)

    model.eval()

    return tokenizer, model

@st.cache_resource
def load_ner_dl():

    with open(WORD2IDX_PATH, "rb") as f:
        word2idx = pickle.load(f)

    with open(IDX2TAG_PATH, "rb") as f:
        idx2tag = pickle.load(f)

    #word2idx = pickle.load(open(WORD2IDX_PATH, "rb"))
    #idx2tag = pickle.load(open(IDX2TAG_PATH, "rb"))

    model = BiLSTM_CRF(len(word2idx), len(idx2tag))
    model.load_state_dict(torch.load(NER_MODEL_PATH, map_location="cpu"))
    model.eval()

    return model, word2idx, idx2tag

@st.cache_resource
def load_ner_pt():

    tokenizer = BertTokenizerFast.from_pretrained(NER_PT_MODEL_PATH)
    model = BertForTokenClassification.from_pretrained(NER_PT_MODEL_PATH)

    # load id2tag
    with open(NER_PT_META_PATH, "rb") as f:
            id2tag = pickle.load(f)

    #id2tag = pickle.load(open(NER_PT_META_PATH, "rb"))

    model.to(device)

    model.eval()

    return tokenizer, model, id2tag

@st.cache_resource
def load_summarizer():

    encoder_model = tf.keras.models.load_model(
        SUMM_ENCODER_PATH,
        compile=False
    )

    decoder_model = tf.keras.models.load_model(
        SUMM_DECODER_PATH,
        compile=False
    )

    with open(SUMM_TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    #tokenizer = pickle.load(open(SUMM_TOKENIZER_PATH, "rb"))

    with open(SUMM_CONFIG_PATH) as f:
        config = json.load(f)

    return encoder_model, decoder_model, tokenizer, config

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_transformer_summarizer():

    tokenizer = AutoTokenizer.from_pretrained(SUMM_PT_PATH)

    model = AutoModelForSeq2SeqLM.from_pretrained(SUMM_PT_PATH)
    model.to(device)
    model.eval()

    return tokenizer, model

ml_model, label_encoder = load_ml()
dl_model, tokenizer_dl = load_dl()
pt_clf_tokenizer, pt_clf_model = load_pt_classifier()
ner_model, word2idx, idx2tag = load_ner_dl()
pt_ner_tokenizer, pt_ner_model, pt_id2tag = load_ner_pt()
encoder_model, decoder_model, summ_tokenizer, summ_config = load_summarizer()
pt_tokenizer, pt_model = load_transformer_summarizer()


# -------------------------------
# NER DL HELPER FUNCTION
# -------------------------------
def ner_predict(text):

    tokens = text.strip().split()

    # Convert to indices
    seq = [word2idx.get(w, word2idx.get("<UNK>", 1)) for w in tokens]

    # Pad
    seq = seq[:NER_MAX_LEN] + [0] * (NER_MAX_LEN - len(seq))

    input_tensor = torch.tensor([seq], dtype=torch.long)

    preds = ner_model(input_tensor)[0]

    tags = [idx2tag[p] for p in preds[:len(tokens)]]

    return tokens, tags

# -------------------------------
# NER PT PREDICTION FUNCTION
# -------------------------------
def ner_pt_predict(text):

    inputs = pt_ner_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = pt_ner_model(**inputs)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=2)[0].cpu().numpy()

    tokens = pt_ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    tags = [pt_id2tag[p] for p in preds]

    return tokens, tags

# ---------------------------------
# SUMMARIZATION EXT.BASELINE.FUNC 
# ---------------------------------
def textrank_summarize(text, top_n=3):

    cleaned = clean_text(text)
    sentences = sent_tokenize(cleaned)

    if len(sentences) <= top_n:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences).toarray()

    sim_matrix = cosine_similarity(vectors)

    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    ranked = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)),
        reverse=True
    )

    summary = " ".join([s for _, s in ranked[:top_n]])

    return summary

# -------------------------------
# SUMMARIZATION DL FUNCTION
# -------------------------------
def generate_summary(text):

    MAX_ART_LEN = summ_config["MAX_ART_LEN"]
    MAX_SUM_LEN = summ_config["MAX_SUM_LEN"]

    cleaned = clean_text(text)

    # Encode input
    seq = summ_tokenizer.texts_to_sequences([cleaned])
    seq = pad_sequences(seq, maxlen=MAX_ART_LEN, padding="post")

    encoder_out, h, c = encoder_model.predict(seq, verbose=0)

    start_token = summ_tokenizer.word_index.get("<sos>")
    end_token = summ_tokenizer.word_index.get("<eos>")

    target_seq = np.array([[start_token]])
    decoded = []

    for _ in range(MAX_SUM_LEN):

        output, h, c = decoder_model.predict(
            [target_seq, encoder_out, h, c],
            verbose=0
        )

        idx = np.argmax(output[0, -1, :])
        word = summ_tokenizer.index_word.get(idx, "")

        if word == "<eos>" or word == "":
            break

        decoded.append(word)

        target_seq = np.array([[idx]])

    return " ".join(decoded)

# -------------------------------
# SUMMARIZATION PT FUNCTION
# -------------------------------

def generate_transformer_summary(text):

    inputs = pt_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():
        summary_ids = pt_model.generate(
            **inputs,
            max_length=180,
            min_length=40,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0
        )

    summary = pt_tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="News Intelligence System", layout="wide")
st.title("📰 Multi-task News Intelligence System")

input_text = st.text_area("Enter News Text", height=200)
st.caption(f"Characters: {len(input_text)}")

col1, col2 = st.columns(2)

with col1:
    task = st.selectbox("Task", ["Classification", "NER", "Summarization"])

with col2:
    if task == "Classification":
        model_choice = st.selectbox(
            "Model",
            ["ML Model", "DL Model", "Pretrained Model"]
        )
    if task == "NER":
        model_choice = st.selectbox(
            "Model",
            ["DL Model", "Pretrained Model"]
        )
    elif task == "Summarization":
        model_choice = st.selectbox(
            "Model",
            ["Extractive Baseline", "DL Model", "Pretrained Transformer Model"]
        )

st.markdown("---")

# -------------------------------
# RUN
# -------------------------------
if st.button("🚀 Run Analysis"):

    if not input_text.strip():
        st.warning("⚠️ Please enter text.")
    else:
        cleaned = clean_text(input_text)

        with st.spinner("Processing... ⏳"):

            # =====================
            # CLASSIFICATION ML MODEL
            # =====================
            if task == "Classification" and model_choice == "ML Model":

                pred = ml_model.predict([cleaned])[0]
                label = label_encoder.inverse_transform([pred])[0]

                st.success("ML Prediction ✅")
                st.write(f" 🧠 Category: {label}")

                # Confidence (only if available)
                try:
                    clf = ml_model.named_steps.get("clf")

                    if hasattr(clf, "predict_proba"):
                        probs = ml_model.predict_proba([cleaned])[0]
                        confidence = probs[pred]
                        st.write(f"Confidence: {confidence:.2%}")
                    else:
                        st.info("Confidence not available for this model")

                except:
                    st.info("Confidence not available")

                log_to_db(
                    user_id="user1",
                    task="Classification",
                    model_family="ML",
                    model_name="logreg_tfidf",
                    input_length=len(input_text),
                    output=label,
                    error=False
                )

            # =====================
            # CLASSIFICATION DL MODEL
            # =====================
            elif task == "Classification" and model_choice == "DL Model":

                seq = tokenizer_dl.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

                probs = dl_model.predict(padded)[0]
                pred = np.argmax(probs)

                label = label_encoder.inverse_transform([pred])[0]

                st.success("DL Prediction ✅")
                st.write(f" 🧠 Category: {label}")
                st.write(f"Confidence: {probs[pred]:.2%}")

                log_to_db(
                    user_id="user1",
                    task="Classification",
                    model_family="DL",
                    model_name="bilstm_glove",
                    input_length=len(input_text),
                    output=label,
                    error=False
                )

            # =====================
            # CLASSIFICATION TRANSFORMER MODEL
            # =====================

            elif task == "Classification" and model_choice == "Pretrained Model":

                inputs = pt_clf_tokenizer(
                    cleaned,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=PT_MAX_LEN
                )

                with torch.no_grad():
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = pt_clf_model(**inputs)

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                pred = int(np.argmax(probs))

                label = pt_clf_model.config.id2label[pred]

                st.success("Transformer Prediction ✅")
                st.write(f"🧠 Category: {label}")
                st.write(f"Confidence: {probs[pred]:.2%}")

                log_to_db(
                    user_id="user1",
                    task="Classification",
                    model_family="Transformer",
                    model_name="bert_classifier",
                    input_length=len(input_text),
                    output=label,
                    error=False
                )

            # =====================
            # NER - DL MODEL
            # =====================
            elif task == "NER" and model_choice == "DL Model":

                tokens, tags = ner_predict(input_text)

                st.success("NER Completed ✅")

                st.subheader("📌 Extracted Entities")

                entities = []
                current_entity = []
                current_tag = None

                for token, tag in zip(tokens, tags):

                    if tag.startswith("B-"):
                        if current_entity:
                            entities.append((" ".join(current_entity), current_tag))
                        current_entity = [token]
                        current_tag = tag[2:]

                    elif tag.startswith("I-") and current_entity:
                        current_entity.append(token)

                    else:
                        if current_entity:
                            entities.append((" ".join(current_entity), current_tag))
                            current_entity = []

                if current_entity:
                    entities.append((" ".join(current_entity), current_tag))

                if entities:
                    for ent, label in entities:
                        st.write(f"**{ent}** → {label}")
                else:
                    st.write("No entities found.")

                log_to_db(
                    user_id="user1",
                    task="NER",
                    model_family="DL",
                    model_name="bilstm_crf",
                    input_length=len(input_text),
                    output=str(entities),
                    error=False
                )

            # =====================
            # NER - TRANSFORMER MODEL
            # =====================
            elif task == "NER" and model_choice == "Pretrained Model":

                tokens, tags = ner_pt_predict(input_text)

                st.success("NER Completed ✅")

                st.subheader("📌 Extracted Entities")

                entities = []
                current_entity = []
                current_tag = None

                for token, tag in zip(tokens, tags):

                    
                    if token in ["[CLS]", "[SEP]", "[PAD]"]:
                        continue

                    
                    if token.startswith("##"):
                        if current_entity:
                            current_entity[-1] += token[2:]
                        continue

                    if tag.startswith("B-"):
                        if current_entity:
                            entities.append((" ".join(current_entity), current_tag))
                        current_entity = [token]
                        current_tag = tag[2:]

                    elif tag.startswith("I-") and current_entity:
                        current_entity.append(token)

                    else:
                        if current_entity:
                            entities.append((" ".join(current_entity), current_tag))
                            current_entity = []
                            current_tag = None

                # Append last entity
                if current_entity:
                    entities.append((" ".join(current_entity), current_tag))


                # Display
                if entities:
                    for ent, label in entities:
                        st.write(f"**{ent}** → {label}")
                else:
                    st.write("No entities found.")

                log_to_db(
                    user_id="user1",
                    task="NER",
                    model_family="Transformer",
                    model_name="bert_ner",
                    input_length=len(input_text),
                    output=str(entities),
                    error=False
                )

            # =====================
            # SUMMARIZATION - EXTRACTIVE
            # =====================
            elif task == "Summarization" and model_choice == "Extractive Baseline":

                summary = textrank_summarize(input_text, top_n=3)

                st.success("Summarization Completed ✅")

                st.subheader("📝 Generated Summary ")
                st.write(summary if summary else "No summary generated.")  

                log_to_db(
                    user_id="user1",
                    task="Summarization",
                    model_family="ML",
                    model_name="textrank",
                    input_length=len(input_text),
                    output=summary,
                    error=False
                )  

            # =====================
            # SUMMARIZATION - DL MODEL
            # =====================
            elif task == "Summarization" and model_choice == "DL Model":

                summary = generate_summary(input_text)

                summary = summary.replace("<unk>", "").strip()
                summary = " ".join(summary.split())

                st.success("Summarization Completed ✅")

                st.subheader("📝 Generated Summary")
                st.write(summary if summary else "No summary generated.")

                log_to_db(
                    user_id="user1",
                    task="Summarization",
                    model_family="DL",
                    model_name="seq2seq",
                    input_length=len(input_text),
                    output=summary,
                    error=False
                )

            # =====================
            # SUMMARIZATION - TRANSFORMER
            # =====================
            elif task == "Summarization" and model_choice == "Pretrained Transformer Model":

                summary = generate_transformer_summary(input_text)

                st.success("Summarization Completed ✅")

                st.subheader("📝 Generated Summary ")
                st.write(summary if summary else "No summary generated.")

                log_to_db(
                    user_id="user1",
                    task="Summarization",
                    model_family="Transformer",
                    model_name="bart_large",
                    input_length=len(input_text),
                    output=summary,
                    error=False
                )
