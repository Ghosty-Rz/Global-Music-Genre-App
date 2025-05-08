from flask import Flask, request, render_template, redirect, url_for
import os
import uuid
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf


# --- Init ---
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load Model ---
model = tf.keras.models.load_model("model\Best_Trained_model_on_MuGen.h5")

# --- Load Genre Labels ---

genre_labels = [
    "Ahidous", "Aita", "andalussi", "Australian Haka", "Khaleeji", "Arabic Pop", "Blues", "Pekinopera", "City pop", "Classical Music",
    "classical tarab", "country", "cumbia", "Salsa", "disco", "EDM", "hiphop", "Bollypop", "Ghazal", "Sufi", "Bandari", "Issawa",
    "jazz", "jpop", "kpop", "Chaabi", "Chgouri", "Dakka Marrakchia", "Gnawa", "mariachi", "merengue", "metal", "Nordic Folk", 
    "Samic Joik", "Pop", "Rai", "Reggae", "RnB", "Rock", "Samba", "Sertanejo",
    "Slavic Folk", "Tango", "Turkish Folk", "Afrobeats", "Griot", "Highlife", "Chanson", "Flamenco", "Opera"
]

# --- Helper: Preprocess Audio ---
def preprocess_audio(file_path, chunk_duration=4, overlap=2, sr=44100, target_shape=(150, 150)):
    y, _ = librosa.load(file_path, sr=sr)
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    
    chunks = []
    for start in range(0, len(y) - chunk_samples + 1, chunk_samples - overlap_samples):
        end = start + chunk_samples
        chunk = y[start:end]
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = tf.image.resize(mel_db[..., np.newaxis], target_shape).numpy()
        chunks.append(mel_db)

    return np.stack(chunks)


@app.route("/", methods=["GET"])
def upload_form():
    return render_template("upload.html")


# --- Route: Upload & Predict ---
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return "No file uploaded", 400

    file = request.files["audio"]
    filename = file.filename
    ext = filename.split(".")[-1]
    temp_filename = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.{ext}")
    file.save(temp_filename)

    try:
        inputs = preprocess_audio(temp_filename)
        predictions = model.predict(inputs)
        average_probs = predictions.mean(axis=0)

        # Top 5 genres
        top_indices = average_probs.argsort()[::-1][:5]
        results = [
            {"genre": genre_labels[i], "confidence": round(float(average_probs[i]) * 100, 2)}
            for i in top_indices
        ]

        return render_template("result.html", filename=filename, main_genre=results[0], others=results[1:])
    except Exception as e:
        return f"Error: {str(e)}", 500
    finally:
        os.remove(temp_filename)

@app.route("/", methods=["GET"])
def home():
    return """
    <h2>Upload a song for genre prediction</h2>
    <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="audio" accept=".wav,.mp3" required>
        <br><br>
        <input type="submit" value="Upload & Predict">
    </form>
    """

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)
