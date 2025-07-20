# 🎙️ Speech Emotion Recognition (SER) with LSTM

This project implements a **real-time Speech Emotion Recognition system** using deep learning. It records audio from the user's microphone, extracts MFCC features, and classifies the emotional state using a **LSTM model** trained on the RAVDESS dataset.

---

## 🚀 Features

- 🎤 Real-time 10-second audio recording from microphone
- 🎧 MFCC feature extraction using Librosa
- 🤖 LSTM model trained on RAVDESS dataset
- 📊 Top 5 emotion predictions with confidence scores
- 💾 Temporary audio storage and cleanup
- ✅ Easy CLI-based interaction

---

## 🎯 Emotions Detected

The model predicts the following emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## 🗂️ Project Structure

Speech_emo/
├── app.py                # Main script to run emotion recognition
├── audio_utils.py        # Audio recording & MFCC feature extraction
├── emotion_utils.py      # Loads model & makes predictions
├── model/
│   └── LSTM_model_*.h5   # Pre-trained LSTM model file
├── temp_audio/           # Stores temporary recorded audio files
└── requirements.txt      # Required dependencies



---

## 🛠️ Setup Instructions

### 🔧 1. Clone Repository


git clone https://github.com/your-username/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install tensorflow==2.10 numpy>=1.24 librosa pyaudio soundfile

▶️ How to Run

python app.py

Sample Output:
[2025-07-19 18:43:11] Top Emotions:
 - angry: 0.84
 - happy: 0.04
 - disgust: 0.12
 - surprised: 0.00
 - sad: 0.00

## 📚 Model Details
Type: Sequential LSTM

Input Shape: (1, 40, 1)

Feature: MFCC (Mean of 40 coefficients)

Trained on: RAVDESS Dataset

Model Used: LSTM_model_Date_Time_2024_01_03_20_39_00___Loss_0.0434___Accuracy_0.9861.h5

## 🙏 Credits
Model & Training: Gaurav Goswami and Dhawal Phalak

Dataset: RAVDESS

Libraries: TensorFlow, Keras, Librosa, PyAudio

## 💡 Future Enhancements
Web-based UI using Streamlit or Flask

Continuous audio emotion tracking

Graphical feedback and history logs

Support for custom datasets


