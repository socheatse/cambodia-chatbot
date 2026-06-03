import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================
# Load configuration first
# ==============================

with open("config.pkl", "rb") as f:
    config = pickle.load(f)

max_len = config["max_len"]

# We need the tokenizer to know how big your vocabulary is for the model structure
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1

# ==============================
# Recreate Architecture & Load Weights
# ==============================

def build_model():
    # Recreating your SimpleRNN architecture shell
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        tf.keras.layers.SimpleRNN(64, return_sequences=True),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])
    return model

@st.cache_resource
def load_my_model():
    model = build_model()
    # Safely load just the numbers (weights) to bypass the Keras .h5 version crash
    model.load_weights("model_weights.weights.h5")
    return model

model = load_my_model()

# ==============================
# Response function
# ==============================

def generate_response(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_len, padding='post')

    pred = model.predict(seq, verbose=0)
    pred = np.argmax(pred, axis=-1)[0]

    result = []
    for idx in pred:
        # Skip the padding index (0) so your chatbot doesn't spit out blank space words
        if idx == 0:
            continue
        for word, index in tokenizer.word_index.items():
            if index == idx:
                result.append(word)
                break

    return " ".join(result) if result else "I'm not sure how to respond to that."


# ==============================
# Streamlit UI
# ==============================

st.title("SimpleRNN Chatbot 🤖")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Input box
user_input = st.text_input("You:", key="input")

# Send button
if st.button("Send"):
    if user_input.strip() != "":
        bot_response = generate_response(user_input)

        # Save to history
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", bot_response))

# Display chat history
st.subheader("Conversation")

for speaker, text in st.session_state.history:
    if speaker == "You":
        st.write(f"🧑 You: {text}")
    else:
        st.write(f"🤖 Bot: {text}")

# ==============================
# End / Clear Chat
# ==============================

col1, col2 = st.columns(2)

with col1:
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []

with col2:
    if st.button("❌ End Chat"):
        st.session_state.history = []
        st.write("Chat ended. Refresh to start again.")
