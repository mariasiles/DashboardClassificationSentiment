# Dashboard interactivo de análisis de sentimiento usando Streamlit
# Funciona con modelo Logistic Regression + TF-IDF
# Muestra resultado principal, probabilidades y palabras influyentes (con expander)

import streamlit as st
import joblib
import sys
import os
import numpy as np

# Ruta para importar funciones de limpieza
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'AnalizarLimpiarDividir')))
from clean_dataset import (
    lowercase_strip, remove_punctuation_space, replace_links, remove_mentions,
    remove_currency, normalize_laughs_en, normalize_repeated_chars, fix_abbr_en,
    remove_special_characters, lemmatize_text
)

# =============================
# Cargar modelo y vectorizador
# =============================
@st.cache(allow_output_mutation=True)
def load_artifacts():
    vectorizer = joblib.load("./VECTORES/tfidf_vectorizer.pkl")
    model = joblib.load("./MODELOS/Ensemble/cache_lr_model.joblib")
    return vectorizer, model

vectorizer, model = load_artifacts()

# =============================
# Función de limpieza de texto
# =============================
def clean_text_pipeline(text):
    text = text.strip()
    text = lowercase_strip(text)
    text = remove_punctuation_space(text)
    text = replace_links(text)
    text = remove_mentions(text)
    text = remove_currency(text)
    text = normalize_laughs_en(text)
    text = normalize_repeated_chars(text)
    text = fix_abbr_en(text)
    text = remove_special_characters(text)
    text = lemmatize_text(text)
    return text

# =============================
# Interfaz
# =============================
st.title("🧠 Twitter Sentiment Analysis")
st.write("Enter a tweet and the model will predict if it is **positive**, **neutral**, or **negative**.")

tweet = st.text_area("✍️ Write your tweet here:")

# =============================
# Predicción
# =============================
if st.button("Analyze sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        # Limpiar el tweet para predicción
        cleaned_tweet = clean_text_pipeline(tweet)

        # Transformar con TF-IDF y predecir
        X = vectorizer.transform([cleaned_tweet])
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        classes = model.classes_

        # =============================
        # Resultado principal visible
        # =============================
        if pred == 'negative':
            st.markdown("<h2 style='color:red;'>😠 NEGATIVE sentiment</h2>", unsafe_allow_html=True)
        elif pred == 'neutral':
            st.markdown("<h2 style='color:gray;'>😐 NEUTRAL sentiment</h2>", unsafe_allow_html=True)
        elif pred == 'positive':
            st.markdown("<h2 style='color:green;'>😊 POSITIVE sentiment</h2>", unsafe_allow_html=True)

        # =============================
        # Probabilidades (expander)
        # =============================
        with st.expander("🔽 Show prediction probabilities"):
            st.write("**Probabilities:**")
            for cls, prob in zip(classes, probs):
                st.write(f"{cls.capitalize()}: {prob:.2f}")

        # =============================
        # Palabras más influyentes (expander)
        # =============================
        with st.expander("🔍 Show top words influencing the prediction"):
            coef_index = list(classes).index(pred)
            coef = model.coef_[coef_index]
            feature_names = vectorizer.get_feature_names_out()
            importance = X.toarray()[0] * coef
            top_indices = importance.argsort()[::-1][:10]
            top_words = [(feature_names[i], importance[i]) for i in top_indices if importance[i] > 0]

            if top_words:
                st.write("**Top words influencing the prediction:**")
                for word, score in top_words:
                    st.write(f"{word} → {score:.4f}")
            else:
                st.write("No influential words found (maybe stopwords or very short tweet).")

# =============================
# Nota
# =============================
st.caption("Model and vectorizer loaded from .joblib files. Input is cleaned with the same pipeline as training. Probabilities and top words are shown.")
