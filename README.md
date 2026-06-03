# Cambodia-Toursim-Chatbot
A simple sequence-to-sequence chatbot that answers common questions about tourism in Cambodia, covering major destinations, local food, and travel logistics. This project was built using TensorFlow/Keras for deep learning and deployed as an interactive web application using Streamlit.
# Demo
Live web application: [Link to your Streamlit app - fill in after deployment]
# Tech Stack 
Language: Python 3.12

- Deep Learning Framework: TensorFlow / Keras (RNN Architecture)
- Data Manipulation: Pandas, NumPy
- Machine Learning Utilities: Scikit-learn
- Deployment & UI: Streamlit
- Visualization: Matplotlib
  
# How to Run Locally 
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `streamlit run app.py`

# Files 
- training_chatbot_TODO_testing.ipynb — Complete Jupyter Notebook containing data exploration, tokenization, preprocessing, model training, and performance plots.
- app.py — Streamlit web interface script for interacting with the chatbot in real-time.
- cambodia_tourism_dataset_large.csv — The dataset containing 2,000 localized tourism question-and-answer pairs.
- model.h5 — The trained and saved recurrent neural network model weights.
- tokenizer.pkl — The saved Keras Tokenizer object used to preserve the exact text-to-integer vocabulary mappings.
- config.pkl - model configuration
- requirements.txt — Python dependencies required to run the environment locally or deploy on Streamlit Cloud.


## Reflections
This project was developed as a case study in foundational deep learning by exploring a basic recurrent neural network (RNN). Originally, the project was started with Simple RNN, which later was switched to an LSTM which uses internal gate to hold onto information much better. However, an analysis of chatbot's performance highlights several limitations: the model faces memory shortage when handling long texts, suffers from conversational amnesia (forgetting past messages in a chat), guesses wildly on off-topic questions, and struggles with unfamiliar words.

