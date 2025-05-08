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

genre_labels = ['Ahidous', 'Aita', 'andalussi', 'AuHaka', 'AwKhaleeji', 'AwPop', 'blues', 'ChPekinopera', 'city pop', 'classical',
 'classical tarab', 'country', 'cumbia', 'CuSalsa', 'disco', 'edm', 'hiphop', 'Inbollypop', 'Inghazal', 'Insufi',
 'IrBandari', 'issawa', 'jazz', 'jpop', 'kpop', 'MaChaabi', 'MaChgouri', 'MaDakkamarrakchia', 'MaGnawa', 'mariachi',
 'merengue', 'metal', 'nrfolk', 'nrsamicjoik', 'pop', 'rai', 'reggae', 'RnB', 'rock', 'samba', 'sertanejo',
 'slavic folk', 'tango', 'turkish folk', 'waafrobeats', 'wagriot', 'wahighlife', 'wechanson', 'weflamenco', 'weopera']


# --- Helper: Preprocess Audio ---
def preprocess_audio(file_path, chunk_duration=4, overlap=2, target_shape=(150, 150)):
    y, sample_rate = librosa.load(file_path, sr=None)
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap * sample_rate)

    chunks = []
    for start in range(0, len(y) - chunk_samples + 1, chunk_samples - overlap_samples):
        end = start + chunk_samples
        chunk = y[start:end]
        mel = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
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
        # Preprocess and predict
        inputs = preprocess_audio(temp_filename)
        print("Chunks shape:", inputs.shape)

        predictions = model.predict(inputs)
        print("Predictions shape:", predictions.shape)

        average_probs = predictions.mean(axis=0)
        print("Average top 5 probs:", np.sort(average_probs)[-5:])

        # Top 5 genres
        top_indices = average_probs.argsort()[::-1][:5]
        print("Average probs:", average_probs)
        print("Top indices:", top_indices)
        print("Genres from labels:")
        for i in top_indices:
            print(f"{i}: {genre_labels[i]}")

        results = [
            {"genre": genre_labels[i], "confidence": round(float(average_probs[i]) * 100, 2)}
            for i in top_indices
        ]

        return render_template("result.html", filename=filename, main_genre=results[0], others=results[1:])
    
    except Exception as e:
        return f"Error: {str(e)}", 500
    
    finally:
        if os.path.exists(temp_filename):
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
