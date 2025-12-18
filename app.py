
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertForSequenceClassification

nltk.download('punkt')
st.title("📧 Spam Email/Text Classifier")

user_input = st.text_area("Enter email/text here:")

@st.cache_resource
def load_models():
    lr_model = joblib.load("logreg_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    scaler = joblib.load("scaler.pkl")
    tokenizer = BertTokenizer.from_pretrained('/content/bert_spam_model')
    bert_model = BertForSequenceClassification.from_pretrained('/content/bert_spam_model')
    bert_model.eval()
    return lr_model, vectorizer, scaler, tokenizer, bert_model

def detect_spam_patterns(text):
    patterns = [r'(?i)fr[3e][3e]', r'(?i)w[i1]n', r'(?i)cl[i1]ck']
    return sum(1 for p in patterns if re.search(p, str(text)))

lr_model, vectorizer, scaler, tokenizer, bert_model = load_models()

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        temp_df = pd.DataFrame([user_input], columns=['text'])
        temp_df['body_length'] = temp_df['text'].apply(lambda x: len(nltk.word_tokenize(str(x))))
        temp_df['pattern_count'] = temp_df['text'].apply(detect_spam_patterns)
        X_temp = np.hstack([temp_df[['pattern_count', 'body_length']].values,
                            vectorizer.transform(temp_df['text']).toarray()])
        X_temp_scaled = scaler.transform(X_temp)
        lr_pred = lr_model.predict(X_temp_scaled)[0]

        enc = tokenizer([user_input], truncation=True, padding=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            output = bert_model(enc['input_ids'], attention_mask=enc['attention_mask'])
        bert_pred = torch.argmax(output.logits, dim=1).item()

        ensemble_score = (0.7 * lr_pred) + (0.3 * bert_pred)
        ensemble_pred = 1 if ensemble_score >= 0.5 else 0

        result = "🚫 Spam" if ensemble_pred == 1 else "✅ Not Spam"
        st.success(f"Prediction: {result}")
