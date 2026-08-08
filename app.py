import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ======================
# Load Model & Files
# ======================
model = load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# ======================
# Next Word Prediction Function
# ======================
def predict_next_word(text):
    
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')
    
    prediction = model.predict(sequence, verbose=0)
    predicted_word_index = np.argmax(prediction, axis=1)[0]

    # Convert index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return ""

# ======================
# Streamlit UI
# ======================
st.title("🧠 LSTM Next Word Predictor")
st.write("Type a sentence and get the predicted next word")

user_input = st.text_input("Enter your text:")

if st.button("Predict"):
    if user_input.strip() != "":
        next_word = predict_next_word(user_input)
        st.success(f"Predicted Next Word: **{next_word}**")
    else:
        st.warning("Please enter some text.")


