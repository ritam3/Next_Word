import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences



def predict_next_word(model, tokenizer, next_words, text, max_sequence_len):
    for _ in range(next_words):
        #convert to token
        token_list = tokenizer.texts_to_sequences([text])[0]
        #path sequences
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        #model prediction
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        # get predict words
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        text += " " + output_word
    return text