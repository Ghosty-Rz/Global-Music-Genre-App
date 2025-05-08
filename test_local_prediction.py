import librosa
import numpy as np
import tensorflow as tf
import os

# Load model
model = tf.keras.models.load_model("model/Best_Trained_model_on_MuGen.h5")

# Your genre labels (same order as during training)
genre_labels = [
    'Ahidous', 'Aita', 'andalussi', 'AuHaka', 'AwKhaleeji', 'AwPop', 'blues', 'ChPekinopera', 'city pop', 'classical',
    'classical tarab', 'country', 'cumbia', 'CuSalsa', 'disco', 'edm', 'hiphop', 'Inbollypop', 'Inghazal', 'Insufi',
    'IrBandari', 'issawa', 'jazz', 'jpop', 'kpop', 'MaChaabi', 'MaChgouri', 'MaDakkamarrakchia', 'MaGnawa', 'mariachi',
    'merengue', 'metal', 'nrfolk', 'nrsamicjoik', 'pop', 'rai', 'reggae', 'RnB', 'rock', 'samba', 'sertanejo',
    'slavic folk', 'tango', 'turkish folk', 'waafrobeats', 'wagriot', 'wahighlife', 'wechanson', 'weflamenco', 'weopera'
]

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

# === TEST A FILE LOCALLY ===
file_path = "Edith Piaf - La foule.wav"  # Replace with your path
predictions = model.predict(preprocess_audio(file_path))
average_probs = predictions.mean(axis=0)

# Print top 5 genres
top_indices = average_probs.argsort()[::-1][:5]
print(predictions)
print("Top Genres:")
for i in top_indices:
    print(f"{genre_labels[i]}: {average_probs[i]*100:.2f}%")
