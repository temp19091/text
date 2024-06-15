import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Define preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# Define emotions and emoji mapping
emotions_emoji_dict = {
    "joy": "😄",
    "sadness": "😢",
    "fear": "😨",
    "anger": "😡",
    "surprise": "😮",
    "neutral": "😐",
    "disgust": "🤢",
    "shame": "😳"
}

# Load the trained model
try:
    with open("text_emotions.pkl", "rb") as f:
        pipe_lr = joblib.load(f)
except FileNotFoundError:
    st.error("Model file 'text_emotions.pkl' not found. Please check the file path.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit app
def main():
    st.title("Text Emotion Detector")
    st.subheader("Detect Emotions in Text")
    
    with st.form(key='my_form'):
        raw_text = st.text_area("Type your text here")
        submit_text = st.form_submit_button(label='Submit')
        
    if submit_text:
        processed_text = preprocess_text(raw_text)
        prediction = pipe_lr.predict([processed_text])[0]
        probability = pipe_lr.predict_proba([processed_text])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("Original Text")
            st.write(raw_text)
            
            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "")
            st.write(f"{prediction}: {emoji_icon}")
            st.write(f"Confidence: {np.max(probability):.2f}")
        
        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]
            
            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions',
                y='probability',
                color='emotions'
            )
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
