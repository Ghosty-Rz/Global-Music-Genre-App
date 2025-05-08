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
    y_pred = model.predict(X_test)  # Shape: (num_chunks, num_genres)

    # 1. Get argmax predictions
    predicted_categories = np.argmax(y_pred, axis=1)

    # 2. Count how often each genre was predicted
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    freq_dict = dict(zip(unique_elements, counts))

    # 3. Average probability across chunks
    avg_probs = np.mean(y_pred, axis=0)  # Shape: (num_genres,)

    # 4. Combine frequency and confidence into a score
    scores = {}
    for genre_idx in range(len(avg_probs)):
        freq = freq_dict.get(genre_idx, 0)
        prob = avg_probs[genre_idx]
        # Weighted score (tune weight as needed)
        scores[genre_idx] = 0.6 * prob + 0.4 * (freq / len(predicted_categories))

    # 5. Sort genres by this score
    sorted_genres = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 6. Select top genres (top 1 + relevant others)
    top_genre = sorted_genres[0][0]
    relevant_others = [idx for idx, score in sorted_genres[1:6]]  # pick up to 5 more

    # 7. Normalize score for display as percentages
    top_scores = [scores[top_genre]] + [scores[i] for i in relevant_others]
    total = sum(top_scores)
    percentages = [(s / total) * 100 for s in top_scores]
    indices = [top_genre] + relevant_others

    return indices, percentages

# Labels
label = [
    "Ahidous", "Aita", "Andalussi", "Australian Haka", "Khaleeji", "Arabic Pop", "Blues", "Pekinopera", "City Pop", "Classical Music", "classical tarab",
    "country", "cumbia", "Salsa", "Disco", "EDM", "Hiphop", "Bollypop", "Ghazal", "Sufi", "Bandari", "issawa",
    "jazz", "Pop", "kpop", "Chaabi", "Chgouri", "Dakka Marrakchia", "Gnawa", "Mariachi", "Merengue", "Metal", "nordic Folk", "Samic Joik", "Pop", "Rai", "Reggae", "RnB", "Rock", "samba", "sertanejo",
    "slavic folk", "tango", "Turkish Folk", "Afrobeats", "Griot", "Highlife", "Chanson", "Flamenco", "Opera"
]


# --- Navigation Header ---
st.markdown("""
    <style>
    /* Make top bar full-width and visible */
    .navbar-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 9999;
        background-color: #ffc34d;
        padding: 15px 40px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Segoe UI', sans-serif;
        border-bottom: 2px solid #f0f0f0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .navbar a {
        margin: 0 15px;
        color: black;
        text-decoration: none;
        font-weight: bold;
    }

    /* Push the content down */
    .main {
        margin-top: 90px;
    }
    </style>

    <div class="navbar-container">
        <div>💿 <a href="#">Home</a> <a href="#">About Us</a></div>
        <div><a href="#" style="background: linear-gradient(90deg, #7b2ff7, #f107a3); padding: 6px 15px; color: white; border-radius: 10px;">TEST A SONG</a></div>
    </div>
""", unsafe_allow_html=True)

# Push content down from the fixed top navbar
st.markdown("<div class='main'>", unsafe_allow_html=True)



# --- Main UI ---
st.title("Let's test your song !")
st.subheader("Upload an MP3 or WAV file")

uploaded_file = st.file_uploader("Choose an MP3 or WAV file", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            uploaded_file.seek(0)  # Reset pointer
            X_test = load_and_preprocess_data(uploaded_file)

            if X_test.size == 0:
                st.error("Something went wrong during audio preprocessing.")
            else:
                # Get top genres and relevance percentages
                indices, percentages = model_prediction(X_test)

                # Your genre label list
                label = [
                "Ahidous", "Aita", "Andalussi", "Australian Haka", "Khaleeji", "Arabic Pop", "Blues", "Pekinopera", "City Pop", "Classical Music", "classical tarab",
                "country", "cumbia", "Salsa", "Disco", "EDM", "Hiphop", "Bollypop", "Ghazal", "Sufi", "Bandari", "issawa",
                "jazz", "Pop", "kpop", "Chaabi", "Chgouri", "Dakka Marrakchia", "Gnawa", "Mariachi", "Merengue", "Metal", "nordic Folk", "Samic Joik", "Pop", "Rai", "Reggae", "RnB", "Rock", "samba", "sertanejo",
                "slavic folk", "tango", "Turkish Folk", "Afrobeats", "Griot", "Highlife", "Chanson", "Flamenco", "Opera"
                ]



                top_genre = label[indices[0]]

                # Top genre display
                st.markdown(f"""
                <div style='text-align: center; margin-top: 40px; margin-bottom: 40px; background-color: #f7f7f7;
                            padding: 30px; border-radius: 25px; box-shadow: 0 0 10px rgba(0,0,0,0.1);'>
                    <h5 style='margin-bottom: 10px;'>{uploaded_file.name}</h5>
                    <h4 style='font-size: 20px;'>Your song falls under:</h4>
                    <h1 style='font-size: 70px; font-weight: bold; color: black;'>{top_genre}</h1>
                </div>
                """, unsafe_allow_html=True)

                # Sub-predictions
                st.markdown("<h5 style='margin-top: 30px;'>But I hear also:</h5>", unsafe_allow_html=True)

                for idx, perc in zip(indices[1:], percentages[1:]):
                    genre = label[idx]
                    st.markdown(f"""
                    <div style='background-color: #f1f1f1; border-radius: 12px; padding: 12px 20px; margin: 10px 0;
                                display: flex; justify-content: space-between; align-items: center; font-size: 18px;'>
                        <div><b>{genre}</b></div>
                        <div><b>{perc:.0f} %</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                st.balloons()