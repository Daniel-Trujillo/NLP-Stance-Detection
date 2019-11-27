import cosine_similarity
import n_gram_matching
import os.path
import pickle
import numpy as np
from preprocessing import DataSet
from sklearn.model_selection import GridSearchCV

COSINE_SIMILARITY_FILE = "../feature_files/similarity_features.pkl"
N_GRAM_FEATURE_FILE = "../feature_files/ngram_matching.pkl"

COSINE_SIMILARITY_TEST_FILE = "../feature_files/similarity_test_features.pkl"
N_GRAM_FEATURE_TEST_FILE = "../feature_files/ngram_matching_test.pkl"

class Models:
    def __init__(self, modelInstance):
        self.model = modelInstance

    def train(self):
        if not os.path.exists(COSINE_SIMILARITY_FILE):
            cs = cosine_similarity.CosineSimilarity()
            cs.create_feature_file(COSINE_SIMILARITY_FILE)

        if not os.path.exists(N_GRAM_FEATURE_FILE):
            ngram = n_gram_matching.NGramMatching()
            ngram.create_feature_file(N_GRAM_FEATURE_FILE)

        cosine_sim_features = np.array(pickle.load(open(COSINE_SIMILARITY_FILE, 'rb'))).reshape(-1, 1)
        n_gram_features = np.array(pickle.load(open(N_GRAM_FEATURE_FILE, 'rb'))).reshape(-1, 1)

        features = np.append(cosine_sim_features, n_gram_features, axis=1)

        ds = DataSet(path="../FNC-1")
        labels = ds.get_labels()

        self.model.fit(features, labels)
        return self.model

    def test(self):
        if not os.path.exists(COSINE_SIMILARITY_TEST_FILE):
            cs = cosine_similarity.CosineSimilarity(name="competition_test")
            cs.create_feature_file(COSINE_SIMILARITY_TEST_FILE)

        if not os.path.exists(N_GRAM_FEATURE_TEST_FILE):
            ngram = n_gram_matching.NGramMatching(name="competition_test")
            ngram.create_feature_file(N_GRAM_FEATURE_TEST_FILE)

        cosine_sim_features = np.array(pickle.load(open(COSINE_SIMILARITY_TEST_FILE, 'rb'))).reshape(-1, 1)
        n_gram_features = np.array(pickle.load(open(N_GRAM_FEATURE_TEST_FILE, 'rb'))).reshape(-1, 1)

        features = np.append(cosine_sim_features, n_gram_features, axis=1)

        test_ds = DataSet(path="../FNC-1", name="competition_test")
        test_labels = test_ds.get_labels()

        return self.model.predict(features), test_labels

    def grid_search(self, model, parameters, k, scoring):
        if not os.path.exists(COSINE_SIMILARITY_FILE):
            cs = cosine_similarity.CosineSimilarity()
            cs.create_feature_file(COSINE_SIMILARITY_FILE)

        if not os.path.exists(N_GRAM_FEATURE_FILE):
            ngram = n_gram_matching.NGramMatching()
            ngram.create_feature_file(N_GRAM_FEATURE_FILE)

        cosine_sim_features = np.array(pickle.load(open(COSINE_SIMILARITY_FILE, 'rb'))).reshape(-1, 1)
        n_gram_features = np.array(pickle.load(open(N_GRAM_FEATURE_FILE, 'rb'))).reshape(-1, 1)

        features = np.append(cosine_sim_features, n_gram_features, axis=1)

        ds = DataSet(path="../FNC-1")
        labels = ds.get_labels()

        clf = GridSearchCV(model, parameters, cv=k, scoring=scoring)

        clf.fit(features, labels)

        mean = clf.cv_results_['mean_test_score'][0]
        std = clf.cv_results_['std_test_score'][0]

        return clf.best_params_, mean, std
