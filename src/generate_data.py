import neurokit2 as nk
import pandas as pd
from src.time_series import TimeSeries


def simulate_ecg(heart_rate=80, noise=.1, duration=60, sampling_rate=128):
    signal = nk.ecg_simulate(duration=duration, noise=noise, heart_rate=heart_rate, sampling_rate=sampling_rate)
    return signal


def simulate_classification_data(class_size=50, heart_rate1=80, heart_rate2=85, noise=.1, duration=60,
                                 sampling_rate=128):
    data = []
    for i in range(class_size * 2):
        if i < class_size:
            heart_rate = heart_rate1
            target = 0
        else:
            heart_rate = heart_rate2
            target = 1

        current = TimeSeries(
            simulate_ecg(
                heart_rate=heart_rate,
                noise=noise,
                duration=duration,
                sampling_rate=sampling_rate
            )
        )

        data.append([current, target])

    data = pd.DataFrame(data, columns=['signal', 'target'])

    return data
