from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import librosa
import os
from tensorflow.image import resize
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = tf.keras.models.load_model("model/Best_Trained_model_on_MuGen.h5")

label = [
                "Ahidous", "Aita", "Andalussi", "Australian Haka", "Khaleeji", "Arabic Pop", "Blues", "Pekinopera", "City Pop", "Classical Music", "classical tarab",
                "country", "cumbia", "Salsa", "Disco", "EDM", "Hiphop", "Bollypop", "Ghazal", "Sufi", "Bandari", "issawa",
                "jazz", "Pop", "kpop", "Chaabi", "Chgouri", "Dakka Marrakchia", "Gnawa", "Mariachi", "Merengue", "Metal", "nordic Folk", "Samic Joik", "Pop", "Rai", "Reggae", "RnB", "Rock", "samba", "sertanejo",
                "slavic folk", "tango", "Turkish Folk", "Afrobeats", "Griot", "Highlife", "Chanson", "Flamenco", "Opera"
                ]

def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration, overlap_duration = 4, 2
    chunk_samples, overlap_samples = chunk_duration * sample_rate, overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    return np.array(data)

def model_prediction(X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    freq_dict = dict(zip(unique_elements, counts))
    avg_probs = np.mean(y_pred, axis=0)
    scores = {i: 0.6 * avg_probs[i] + 0.4 * (freq_dict.get(i, 0) / len(predicted_categories)) for i in range(len(avg_probs))}
    sorted_genres = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_genre = sorted_genres[0][0]
    relevant_others = [idx for idx, _ in sorted_genres[1:6]]
    top_scores = [scores[top_genre]] + [scores[i] for i in relevant_others]
    total = sum(top_scores)
    percentages = [(s / total) * 100 for s in top_scores]
    indices = [top_genre] + relevant_others
    return [label[i] for i in indices], percentages

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        X_test = load_and_preprocess_data(filepath)
        genres, percentages = model_prediction(X_test)

        return render_template("result.html", filename=filename, top=genres[0], others=zip(genres[1:], percentages[1:]))

    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
