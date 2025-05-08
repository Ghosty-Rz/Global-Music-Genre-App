import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from matplotlib import pyplot
from tensorflow.image import resize

# --- Load Global Model ---
@st.cache_resource()
def load_global_model():
    return tf.keras.models.load_model("model/Best_Trained_model_on_MuGen.h5")

# --- Preprocessing ---
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    return np.array(data)

# --- Fallback Prediction Function (Global Only, Smart Filtering) ---
def fallback_prediction(X_test):
    model = load_global_model()
    pred = model.predict(X_test)  # (n_chunks, 50)
    avg_probs = np.mean(pred, axis=0)

    threshold = 0.10
    relevant_indices = [i for i, p in enumerate(avg_probs) if p > threshold]
    scores = [avg_probs[i] for i in relevant_indices]
    total = sum(scores)
    percentages = [(s / total) * 100 for s in scores]

    return relevant_indices, percentages

# --- Streamlit App ---
st.title("🎵 Music Genre Classification")
st.subheader("Upload an MP3 or WAV file")

uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            uploaded_file.seek(0)
            X_test = load_and_preprocess_data(uploaded_file)

            if X_test.size == 0:
                st.error("Something went wrong during audio preprocessing.")
            else:
                indices, percentages = fallback_prediction(X_test)

                label = [
                    "Ahidous", "Aita", "Andalussi", "Australian Haka", "Khaleeji", "Arabic Pop", "Blues", "Pekinopera", "City Pop", "Classical Music", "classical tarab",
                    "country", "cumbia", "Salsa", "Disco", "EDM", "Hiphop", "Bollypop", "Ghazal", "Sufi", "Bandari", "issawa",
                    "jazz", "Pop", "kpop", "Chaabi", "Chgouri", "Dakka Marrakchia", "Gnawa", "Mariachi", "Merengue", "Metal", "nordic Folk", "Samic Joik", "Pop", "Rai", "Reggae", "RnB", "Rock", "samba", "sertanejo",
                    "slavic folk", "tango", "Turkish Folk", "Afrobeats", "Griot", "Highlife", "Chanson", "Flamenco", "Opera"
                ]

                st.success(f"🎧 Top Prediction: **{label[indices[0]]}**")
                st.write("### 🔍 Relevant Genre Breakdown:")
                for idx, perc in zip(indices, percentages):
                    st.write(f"{label[idx]}: {perc:.2f}%")

                st.bar_chart({label[idx]: perc for idx, perc in zip(indices, percentages)})
                st.balloons()
