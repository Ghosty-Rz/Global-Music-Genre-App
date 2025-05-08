import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from matplotlib import pyplot
import numpy as np
from tensorflow.image import resize

#Function
@st.cache_resource()
def load_model():
  model = tf.keras.models.load_model("model\Best_Trained_model_on_MuGen.h5")
  return model


# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
                
    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    # Iterate over each chunk
    for i in range(num_chunks):
                    # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
                    
                    # Extract the chunk of audio
        chunk = audio_data[start:end]
                    
                    # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                    
                #mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)



#Tensorflow Model Prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred,axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    #print(unique_elements, counts)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]


# STREAMLIT UI #
st.title("🎵 Music Genre Classification")
st.subheader("Upload an MP3 or WAV file")

uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            X_test = load_and_preprocess_data(uploaded_file)
            pred_index = model_prediction(X_test)
            label = [
            "Ahidous", "Aita", "Andalussi", "Australian Haka", "Khaleeji", "Arabic Pop", "Blues", "Pekinopera", "City Pop", "Classical Music", "classical tarab",
            "country", "cumbia", "Salsa", "Disco", "EDM", "Hiphop", "Bollypop", "Ghazal", "Sufi", "Bandari", "issawa",
            "jazz", "Pop", "kpop", "Chaabi", "Chgouri", "Dakka Marrakchia", "Gnawa", "Mariachi", "Merengue", "Metal", "nordic Folk", "Samic Joik", "Pop", "Rai", "Reggae", "RnB", "Rock", "samba", "sertanejo",
            "slavic folk", "tango", "Turkish Folk", "Afrobeats", "Griot", "Highlife", "Chanson", "Flamenco", "Opera"
            ]
            st.success(f"It's a **{label[pred_index]}** track! 🎧")
            st.balloons()

    


