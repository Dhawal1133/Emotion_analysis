import librosa
import numpy as np
import soundfile as sf
import os
import uuid

TEMP_AUDIO_DIR = "temp_audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

SAMPLE_RATE = 22050
DURATION = 10  # seconds

def extract_features(audio_path):
    """
    Extract only MFCC features and reshape to (1, 40, 1)
    as required by Billy's LSTM model.
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # shape (40,)
    return mfcc_mean.reshape(1, 40, 1)

def is_silent(audio_path, threshold=0.005):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    energy = np.sqrt(np.mean(y**2))
    return energy < threshold

def save_temp_audio(data, samplerate=SAMPLE_RATE):
    filename = os.path.join(TEMP_AUDIO_DIR, f"{uuid.uuid4()}.wav")
    sf.write(filename, data, samplerate)
    return filename
