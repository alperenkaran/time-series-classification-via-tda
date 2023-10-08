from ripser import ripser
import numpy as np
import gudhi
from gtda.time_series import SingleTakensEmbedding
from src.persistence_diagram import PersistenceDiagram


class Subwindow:
    def __init__(self, subwindow):
        self.subwindow = subwindow

    def get_level_set_diagrams(self):
        return {
            'lower': self._level_set_diagram(self.subwindow, is_lower=True),
            'upper': self._level_set_diagram(self.subwindow, is_lower=False)
        }

    def get_embedding_diagrams(self, embedding_dimension):
        embedder = SingleTakensEmbedding(parameters_type="fixed", dimension=embedding_dimension)
        data = embedder.fit_transform(self.subwindow)
        diagrams = ripser(data)['dgms']
        diagrams = {
            'homology0': diagrams[0],
            'homology1': diagrams[1]
        }
        return diagrams

    @staticmethod
    def _level_set_diagram(x, is_lower=True):
        if not is_lower:
            x *= -1

        # noinspection PyUnresolvedReferences
        st = gudhi.SimplexTree()

        for i, value in enumerate(x):
            st.insert([i], filtration=value)

        for i in range(len(x) - 1):
            st.insert([i, i + 1], filtration=max(x[i], x[i + 1]))

        lower_level_set_persistence = st.persistence(homology_coeff_field=2)

        diagram = np.array([x[1] for x in lower_level_set_persistence])

        # if not is_lower:
        #     diagram *= -1

        return diagram

    def get_all_diagrams(self, embedding_dimensions=None):
        if not embedding_dimensions:
            embedding_dimensions = [3]

        if isinstance(embedding_dimensions, int):
            embedding_dimensions = [embedding_dimensions]

        diagrams = []

        level_set_diagrams = self.get_level_set_diagrams()
        diagrams.append(level_set_diagrams['lower'])
        diagrams.append(level_set_diagrams['upper'])

        for dim in embedding_dimensions:
            embedding_diagrams = self.get_embedding_diagrams(dim)
            diagrams.append(embedding_diagrams['homology0'])
            diagrams.append(embedding_diagrams['homology1'])

        return diagrams

    def get_features(self, embedding_dimensions=None):
        features = []
        for diagram in self.get_all_diagrams(embedding_dimensions):
            diagram = PersistenceDiagram(diagram)
            this_features = diagram.get_features().values()
            features.extend(this_features)

        return np.array(features)
