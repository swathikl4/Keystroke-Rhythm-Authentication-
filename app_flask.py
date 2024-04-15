from flask import Flask, render_template, request, redirect, url_for

import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model('keystroke_rhythm_prediction_model.h5')

# Load the label encoder classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

def predict_subjects(prediction_data):
    # Reshape the input data for LSTM
    prediction_data_lstm = prediction_data.values.reshape((prediction_data.shape[0], 1, prediction_data.shape[1]))
    
    # Predict classes
    predictions_prob = model.predict(prediction_data_lstm)
    predictions = np.argmax(predictions_prob, axis=1)
    
    # Map predictions to subjects using LabelEncoder
    predicted_subjects = label_encoder.inverse_transform(predictions)
    return predicted_subjects

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Load the data
            data = pd.read_csv(file)
            
            # Check if H.period exceeds 1
            if (data['H.period'] > 1).any():
                return redirect(url_for('error'))
            
            # Predict subjects
            predicted_subjects = predict_subjects(data)
            
            return render_template('predict.html', predicted_subjects=predicted_subjects)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Load the data
            data = pd.read_csv(file)
            
            # Check if H.period exceeds 1
            if (data['H.period'] > 1).any():
                return redirect(url_for('error'))
            
            # Predict subjects
            predicted_subjects = predict_subjects(data)
            
            return render_template('predict.html', predicted_subjects=predicted_subjects)
    return render_template('index.html')

@app.route('/error')
def error():
    return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)
