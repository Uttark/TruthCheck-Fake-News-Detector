import gradio as gr
import pickle
import re
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import os
import zipfile

# ============ NLTK Setup ============

# Define the local path where nltk_data is/will be stored
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')

# Unzip the NLTK data (like stopwords, punkt) if it doesn't exist yet
if not os.path.exists(nltk_data_path):
    with zipfile.ZipFile('nltk_data.zip', 'r') as zip_ref:
        zip_ref.extractall(nltk_data_path)

# Add the extracted NLTK path to nltk's data search paths
nltk.data.path.append(nltk_data_path)

# ============ Load Models and Tokenizers ============

# Load the pre-trained Logistic Regression model
with open("logreg_model.pkl", "rb") as f:
    logreg_model = pickle.load(f)

# Load the pre-trained Naive Bayes model
with open("nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

# Load the saved TF-IDF vectorizer used during model training
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Load the tokenizer used to preprocess text for the GloVe model
with open("glove_tokenizer.pkl", "rb") as f:
    glove_tokenizer = pickle.load(f)

# Load the pre-trained GloVe-based deep learning model
model_glove = tf.keras.models.load_model("glove_model.h5")

# ============ Constants and Preprocessing Tools ============

# Max sequence length for the GloVe model input
MAX_LENGTH = 300

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Initialize a lemmatizer for word normalization
lemmatizer = WordNetLemmatizer()

# ============ Text Cleaning Function ============

def clean_text(text):
    """
    Cleans the input text by removing punctuation, stopwords, numbers, and performing lemmatization.
    """
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.replace('“', '').replace('”', '').replace("’", "'").replace("‘", "'")
    text = re.sub(r"'s\b", '', text)

    # Tokenize and lemmatize words while removing stopwords
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# ============ Prediction Logic ============

def predict_ensemble(text):
    """
    Runs all three models on the cleaned input text and returns an ensemble prediction.
    """
    cleaned = clean_text(text)

    # Return early if input is too short to be meaningful
    if len(cleaned.strip()) <= 10:
        return "Input too short to analyze."

    # TF-IDF vectorization for classical ML models
    tfidf_vec = tfidf_vectorizer.transform([cleaned])
    
    # Get probability predictions from each model
    prob_nb = nb_model.predict_proba(tfidf_vec)[0][1]          # Naive Bayes
    prob_logreg = logreg_model.predict_proba(tfidf_vec)[0][1]  # Logistic Regression

    # Tokenize and pad text for the deep learning model
    glove_seq = glove_tokenizer.texts_to_sequences([cleaned])
    glove_pad = pad_sequences(glove_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    prob_glove = model_glove.predict(glove_pad)[0][0]          # GloVe + Deep Learning

    # Ensemble: weighted average of the individual model predictions
    ensemble_score = 0.50 * prob_nb + 0.10 * prob_logreg + 0.40 * prob_glove

    # Classify based on ensemble score threshold
    label = "✅ Real News" if ensemble_score >= 0.47 else "❌ Fake News"

    # Format the result with individual model scores
    explanation = f"""
**Model 1 (Naive Bayes):** {prob_nb:.4f}  
**Model 2 (Logistic Regression):** {prob_logreg:.4f}  
**Model 3 (GloVe + DL):** {prob_glove:.4f}  
**Ensemble Score:** {ensemble_score:.4f}  
**Final Prediction:** {label}
"""
    return explanation

# ============ Gradio UI ============

# Define the Gradio interface for the web app
interface = gr.Interface(
    fn=predict_ensemble,  # Function to call on submission
    inputs=gr.Textbox(lines=8, placeholder="Paste your news article here...", label="News Article"),
    outputs=gr.Markdown(label="Prediction"),
    title="📰 Fake News Detector",
    description="This tool uses 3 models (Naive Bayes, Logistic Regression, GloVe-based Deep Learning) to classify news as real or fake using an ensemble method.",
    allow_flagging="never"
)

# Launch the app when the file is run directly
if __name__ == "__main__":
    interface.launch()
