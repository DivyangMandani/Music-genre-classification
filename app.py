import os
import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('dl_model.h5')  

# Function to extract spectrogram from audio file and convert it to an image-like format
def extract_spectrogram(audio_path, target_size=(224, 224)):
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        # Convert the spectrogram to dB scale
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        # Resize the image to the target size
        spectrogram_image_resized = tf.image.resize(np.expand_dims(spectrogram_db, axis=-1), target_size)
        return spectrogram_image_resized.numpy()
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Function to classify genre
def classify_genre(audio_path):
    features = extract_spectrogram(audio_path)
    if features is not None:
        features_rgb = np.concatenate([features, features, features], axis=-1)
        input_data = np.expand_dims(features_rgb, axis=0)
        prediction = model.predict(input_data)
        print(prediction)
        predicted_class = np.argmax(prediction)
        return predicted_class
    else:
        return None

# Streamlit app
st.title("Music Genre Classification App")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file (MP3 or WAV)", type=["mp3", "wav"])
category_labels=['disco',
 'metal',
 'reggae',
 'blues',
 'rock',
 'classical',
 'jazz',
 'hiphop',
 'country',
 'pop']

if uploaded_file is not None:
    # Save the uploaded file locally
    audio_path = "uploaded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Classify genre
    predicted_class = classify_genre(audio_path)

    if predicted_class is not None:
        # Get the genre label based on the index
        genre_label = category_labels[predicted_class]
        st.success(f"Predicted Genre: {genre_label}")
    else:
        st.error("Error processing the audio file. Please try again.")
