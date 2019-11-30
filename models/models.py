import cosine_similarity
import n_gram_matching
import numpy as np
from preprocessing import DataSet
from sklearn.model_selection import GridSearchCV


class Models:
    def __init__(self, modelInstance):
        self.model = modelInstance
        features = [cosine_similarity.CosineSimilarity(), n_gram_matching.NGramMatching()]
        self.features_train = np.hstack([feature.read() for feature in features])
        self.labels_train = DataSet(path="../FNC-1").get_labels()
        self.features_test = np.hstack([feature.read('test') for feature in features])
        self.labels_test = DataSet(path="../FNC-1", name="competition_test").get_labels()

    def train(self):
        self.model.fit(self.features_train, self.labels_train)
        return self.model

    def test(self):
        return self.model.predict(self.features_test), self.labels_test

    def grid_search(self, parameters, k, scoring):
        clf_cv = GridSearchCV(self.model, param_grid=parameters, cv=k, scoring=scoring)
        clf_cv.fit(self.features_train, self.labels_train)
        mean = clf_cv.cv_results_['mean_test_score'][0]
        std = clf_cv.cv_results_['std_test_score'][0]
        return clf_cv.best_params_, mean, std
