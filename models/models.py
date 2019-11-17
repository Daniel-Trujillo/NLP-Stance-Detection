from sklearn.linear_model import LogisticRegression
import cosine_similarity
import n_gram_matching
import os.path
import pickle
import numpy as np
from preprocessing import DataSet

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

        cosine_sim_features = np.array(pickle.load(COSINE_SIMILARITY_FILE)).reshape(-1, 1)
        n_gram_features = np.array(pickle.load(N_GRAM_FEATURE_FILE)).reshape(-1, 1)

        features = np.append(cosine_sim_features, n_gram_features, axis=1)

        ds = DataSet()
        labels = ds.get_labels()

        self.model.fit(features, labels)
        return self.model

    def test(self):
        if not os.path.exists(COSINE_SIMILARITY_FILE):
            cs = cosine_similarity.CosineSimilarity(name="competition_test")
            cs.create_feature_file(COSINE_SIMILARITY_FILE)

        if not os.path.exists(N_GRAM_FEATURE_FILE):
            ngram = n_gram_matching.NGramMatching(name="competition_test")
            ngram.create_feature_file(N_GRAM_FEATURE_FILE)

        cosine_sim_features = np.array(pickle.load(COSINE_SIMILARITY_FILE)).reshape(-1, 1)
        n_gram_features = np.array(pickle.load(N_GRAM_FEATURE_FILE)).reshape(-1, 1)

        features = np.append(cosine_sim_features, n_gram_features, axis=1)

        test_ds = DataSet(name="competition_test")
        test_labels = test_ds.get_labels()

        return self.model.predict(features, test_labels)