import numpy as np
from gtda.time_series import SingleTakensEmbedding
from src.subwindow import Subwindow
from tqdm import tqdm


class TimeSeries:
    def __init__(self, time_series):
        self.time_series = time_series
        self.subwindows = None

    def compute_subwindows(self, window_size, window_shift):
        windower = SingleTakensEmbedding(
            parameters_type="fixed",
            dimension=window_size,
            stride=window_shift
        )
        self.subwindows = windower.fit_transform(self.time_series)

    def get_features(self, subwindow_size, subwindow_shift, embedding_dimensions=None):
        if not self.subwindows:
            self.compute_subwindows(subwindow_size, subwindow_shift)

        list_of_subwindow_features = []

        for subwindow in tqdm(self.subwindows):
            subwindow = Subwindow(subwindow)
            subwindow_features = subwindow.get_features(embedding_dimensions)
            list_of_subwindow_features.append(subwindow_features)

        features_avg = np.average(list_of_subwindow_features, axis=0)
        features_std = np.std(list_of_subwindow_features, axis=0)

        return np.concatenate([features_avg, features_std])
