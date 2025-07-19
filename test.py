from keras.models import load_model
model = load_model("model/LSTM_model_Date_Time_2024_01_03_20_39_00___Loss_0.043416813015937805___Accuracy_0.9861111044883728.h5")
print(features.shape)  # Should print: (1, 40, 1)
