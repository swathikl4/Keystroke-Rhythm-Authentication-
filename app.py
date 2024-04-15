import streamlit as st
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

def main():
    st.title('Keystroke Rhythm Prediction App using LSTM')
    st.write('Upload a CSV file with keystroke rhythm data to predict the subject.')
    
    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    
    if uploaded_file is not None:
        # Load the data
        data = pd.read_csv(uploaded_file)
        
        # Display the uploaded data
        st.write('Uploaded data:')
        st.write(data)
        
        # Predict subjects
        predicted_subjects = predict_subjects(data)
        
        # Display the predicted subjects
        st.write('Predicted user is :')
        st.write(predicted_subjects)

if __name__ == '__main__':
    main()
