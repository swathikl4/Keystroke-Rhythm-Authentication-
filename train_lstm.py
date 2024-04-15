
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.inspection import permutation_importance

# Load the dataset
data = pd.read_csv('DSL-StrongPasswordData.csv')

# Encode the subject column
label_encoder = LabelEncoder()
data['subject_encoded'] = label_encoder.fit_transform(data['subject'])

# Save the LabelEncoder classes
np.save('label_encoder_classes.npy', label_encoder.classes_)

# Define the input features and target variable
X = data.drop(['subject', 'subject_encoded'], axis=1)
y = data['subject_encoded']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data for LSTM
X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(100, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dense(56, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Initialize lists to store metrics
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []

# Fit the model
for epoch in range(1, 61):
    history = model.fit(X_train_lstm, y_train, epochs=1, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=0)
    
    # Store training and validation loss
    training_loss.append(history.history['loss'][0])
    validation_loss.append(history.history['val_loss'][0])
    
    # Store training and validation accuracy
    training_accuracy.append(history.history['accuracy'][0])
    validation_accuracy.append(history.history['val_accuracy'][0])
    
    print(f'Epoch {epoch}/{60}')

# Save the model
model.save('keystroke_rhythm_prediction_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test_lstm, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(range(1, 61), training_loss, label='Training Loss')
plt.plot(range(1, 61), validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(range(1, 61), training_accuracy, label='Training Accuracy')
plt.plot(range(1, 61), validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.show()

# Generate predictions
predictions_prob = model.predict(X_test_lstm)
predictions = np.argmax(predictions_prob, axis=1)


# Confusion matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(12, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(label_encoder.classes_))
plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
plt.yticks(tick_marks, label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification report
print('Classification Report:')
print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

