from src.utils import get_run_args
from src.generate_data import simulate_ecg, simulate_classification_data
from src.time_series import TimeSeries
from src.classification import train_simple_model_and_print_results


def main():
    run_mode, = get_run_args(['run_mode', 'default'])

    if run_mode == 'classification_example':
        data = simulate_classification_data(heart_rate1=80, heart_rate2=85)

        data['features'] = data['signal'].map(
            lambda x: x.get_features(subwindow_size=512, subwindow_shift=256, embedding_dimensions=[32, 64, 128])
        )

        train_simple_model_and_print_results(data['features'], data['target'])

    elif run_mode == 'feature_engineering_example':
        ecg_signal = TimeSeries(simulate_ecg(heart_rate=80))
        features = ecg_signal.get_features(
            subwindow_size=128 * 4,  # 4 seconds
            subwindow_shift=128 * 2,  # 2 seconds
            embedding_dimensions=[32, 64, 128]
        )
        return features


if __name__ == '__main__':
    main()
