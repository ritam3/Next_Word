import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from predict import predict_next_word

#Load the LSTM Model
model=load_model('model_file/my_model.keras')

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")

input_text=st.text_input("Enter the sequence of Words","Sherlock was")

num_words = st.number_input(
    "Select the number of words:",
    min_value=1,  
    max_value=10, 
    value=10,  
    step=1
)

if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model,tokenizer, num_words, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')

