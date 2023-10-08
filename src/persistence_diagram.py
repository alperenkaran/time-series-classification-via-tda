import numpy as np
from gtda.diagrams import BettiCurve, PersistenceLandscape


class PersistenceDiagram:
    def __init__(self, values):
        values = values[~np.isinf(values[:, 1])]  # remove inf
        self.values = values
        self.lives = values[:, 1] - values[:, 0]
        self.min_birth = min(self.values[:, 0])
        self.max_death = max(self.values[:, 1])
        self.gtda_diagram = self._get_gtda_diagram()
        self.betti_curve = None
        self.persistence_landscape = None

    def get_features(self):
        return {
            'bottleneck_amplitude': self.bottleneck_amplitude(),
            'wasserstein_amplitude1': self.wasserstein_amplitude(degree=1),
            'wasserstein_amplitude2': self.wasserstein_amplitude(degree=2),
            'persistent_entropy': self.persistent_entropy(),
            'betti_l1norm': self.get_betti_curve_area(norm_degree=1),
            'betti_l2norm': self.get_betti_curve_area(norm_degree=2),
            'landscape_l1norm': self.get_landscape_area(norm_degree=1),
            'landscape_l2norm': self.get_landscape_area(norm_degree=2)
        }

    def bottleneck_amplitude(self):
        return max(self.lives)

    def wasserstein_amplitude(self, degree):
        return sum(self.lives ** degree) ** (1 / degree)

    def _get_gtda_diagram(self):
        # giotto-tda library requires a 3d array with 3 columns
        dgm = self.values
        dgm = np.concatenate((dgm, np.zeros((dgm.shape[0], 1))), axis=1)
        return dgm.reshape(1, *dgm.shape)

    def _set_betti_curve(self):
        self.betti_curve = BettiCurve().fit_transform(self.gtda_diagram)[0][0]

    def _set_landscape(self):
        self.persistence_landscape = PersistenceLandscape().fit_transform(self.gtda_diagram)[0][0]

    def get_betti_curve_area(self, norm_degree):
        if self.betti_curve is None:
            self._set_betti_curve()

        return self.get_function_norm(self.betti_curve, self.min_birth, self.max_death, norm_degree)

    def get_landscape_area(self, norm_degree):
        if self.persistence_landscape is None:
            self._set_landscape()

        return self.get_function_norm(self.persistence_landscape, self.min_birth, self.max_death, norm_degree)

    def persistent_entropy(self):
        normalized_lives = self.lives / sum(self.lives)
        return - sum(normalized_lives * np.log(normalized_lives))

    @staticmethod
    def get_function_norm(function_values, domain_start, domain_end, norm_degree):
        step_size = (domain_end - domain_start) / (len(function_values) - 1)
        lp_norm = np.sum(np.abs(function_values) ** norm_degree) ** (1 / norm_degree) * step_size
        return lp_norm
