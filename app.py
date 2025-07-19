import pyaudio
import numpy as np
import time
import os
from datetime import datetime

from audio_utils import extract_features, save_temp_audio, is_silent
from emotion_utils import predict_emotion

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050
RECORD_SECONDS = 10

print("[INFO] Starting 10-second recording... Speak now!")

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

frames = []
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(np.frombuffer(data, dtype=np.float32))

stream.stop_stream()
stream.close()
p.terminate()

# Save audio
audio_data = np.hstack(frames)
wav_path = save_temp_audio(audio_data)

# Check silence
if is_silent(wav_path):
    print("[INFO] Silence detected. No speech to analyze.")
    exit()

# Extract features & predict emotion
features = extract_features(wav_path)  # shape will be (1, 40)
top_emotions = predict_emotion(features)


# Output
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{timestamp}] Top Emotions:")
for emo, score in top_emotions:
    print(f" - {emo}: {score:.2f}")
# Clean up temporary audio file
os.remove(wav_path)