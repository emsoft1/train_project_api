import joblib
import pandas as pd
from keras.models import load_model

THRESHOLD = 0.19
sample_data = pd.read_csv("train_model/deploy_fourier_model/ui_data_exp2_1024_buckets.csv")
user_options = sample_data.timestamp.to_list()
sample_data.set_index('timestamp', inplace=True)
#model = joblib.load("scaler_and_ft_exp2_1024b.mod")
model = load_model("train_model/deploy_fourier_model/ft_exp2_1024b.h5")
scaler = joblib.load("train_model/deploy_fourier_model/scaler.mod")


# import data needed to plot average vibrations
averaged_amplitude_file = "train_model/deploy_fourier_model/vibration_data.csv"
df_averaged_amplitude = pd.read_csv(averaged_amplitude_file)
df_averaged_amplitude.set_index('timestamp',inplace=True)

def get_user_options():
    return user_options

def predict(time_stamp):
    test_signal = sample_data.loc[time_stamp,:]
    signal_as_np = test_signal.to_numpy().reshape((1,-1))
    scaled_signal = scaler.transform(signal_as_np)

    prediction = model.predict(scaled_signal)
    #original_signal_scaled = scaler.transform(signal_as_np)

    loss = ((prediction - scaled_signal)**2).mean()

    if loss < THRESHOLD:
        return False
    return True

def get_signal_for_plotting(time_stamp):
    test_signal = sample_data.loc[time_stamp,:]
    signal_as_np = test_signal.to_numpy().reshape((1,-1))
    return signal_as_np

def get_vibration_dataframe():
    return df_averaged_amplitude

# if __name__ == '__main__':
#     out = []
#     for time in get_user_options():
#         out.append(predict(time))
#     print(out)
