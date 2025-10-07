from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

model = tf.keras.models.load_model("lstm_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)

max_length = 14  

def generate_text(seed_text):
    token_text = tok.texts_to_sequences([seed_text])[0]
    token_text = pad_sequences([token_text], maxlen=max_length, padding='pre')
    pred = np.argmax(model.predict(token_text, verbose=0))
    next_word = tok.index_word.get(pred, "")
    return next_word

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['text']
    result = generate_text(data)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)