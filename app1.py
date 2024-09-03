from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
from tensorflow import keras

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Ensure this folder exists

# Load your trained model (adjust the path as necessary)
model = keras.models.load_model('saved_model.keras')
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # Adjust if needed

@app.route('/')
def home():
    return render_template('index.html')

def predict_emotion(file_path):
    try:
        print(f"Attempting to load file: {file_path}")  # Log the file path
        signal, sr = librosa.load(file_path, sr=22050)

        # Process the signal to extract MFCC features
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfccs = np.mean(mfccs.T, axis=0)
        mfccs = mfccs.reshape(1, -1)

        # Make a prediction
        prediction = model.predict(mfccs)
        predicted_emotion = emotions[np.argmax(prediction)]
        return predicted_emotion
    except Exception as e:
        print(f"Error predicting emotion: {str(e)}")  # Detailed error output
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Ensure the uploads directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Save the file to the uploads directory
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)  # Save the file

    print(f"Received file: {file_path}")  # Log the uploaded file path

    # Predict the emotion
    predicted_emotion = predict_emotion(file_path)

    # Clean up by deleting the file
    # os.remove(file_path)

    if predicted_emotion is None:
        return jsonify({'error': 'An error occurred while predicting the emotion.'})

    return jsonify({'emotion': predicted_emotion})

if __name__ == '__main__':
    app.run(debug=True)
