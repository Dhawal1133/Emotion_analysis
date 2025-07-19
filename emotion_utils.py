
from keras.models import load_model
import numpy as np



model = load_model("model/LSTM_model_Date_Time_2024_01_03_20_39_00___Loss_0.043416813015937805___Accuracy_0.9861111044883728.h5", compile=False)



class_labels = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']


def predict_emotion(features):
    # Don't expand dims if already in (1, 40, 1) shape
    pred = model.predict(features)[0]
    return sorted({label: float(prob) for label, prob in zip(class_labels, pred)}.items(), key=lambda x: x[1], reverse=True)[:5]



