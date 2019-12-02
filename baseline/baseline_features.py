import numpy as np
from preprocessing import DataSet
import pickle
import os
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
import nltk
nltk.download('vader_lexicon')

BASELINE_FILE = "../feature_files/baseline_features.pkl"
BASELINE_TEST_FILE = "../feature_files/baseline_test_features.pkl"


class BaselineFeature:
    def __init__(self, path="../FNC-1/", name="train"):
        self.path = path
        self.name = name

    def get_feature(self):
        dataset = DataSet(path=self.path, name=self.name)
        h, b = [], []
        stances = dataset.stances
        for stance in stances:
            h.append(stance['Headline'])
            b.append(dataset.articles[stance['Body ID']])

        X_overlap = word_overlap_features(h, b)
        X_refuting = refuting_features(h, b)
        X_polarity = polarity_features(h, b)
        X_hand = hand_features(h, b)

        X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
        return X

    def create_feature_file(self, path):
        features = self.get_feature()

        with open(path, 'wb') as f:
            pickle.dump(features, f)

    def read(self, name="train"):
        self.name = name
        if name == 'train':
            if not os.path.exists(BASELINE_FILE):
                self.create_feature_file(BASELINE_FILE)
            return np.array(pickle.load(open(BASELINE_FILE, 'rb')))
        else:
            if not os.path.exists(BASELINE_TEST_FILE):
                self.create_feature_file(BASELINE_TEST_FILE)
            return np.array(pickle.load(open(BASELINE_TEST_FILE, 'rb')))