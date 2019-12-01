import sklearn
import pickle
from preprocessing import DataSet
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

SVD_FILE = "../feature_files/svd_features.pkl"
SVD_TEST_FILE = "../feature_files/svd_test_features.pkl"


class SVD:
    def __init__(self, name="train", path="../FNC-1", lemmatize=True, remove_stop=True, remove_punc=False, sent=True):
        self.path = path
        self.name = name
        self.lemmatize = lemmatize
        self.remove_stop = remove_stop
        self.remove_punc = remove_punc
        self.sent = sent

    def get_feature(self):
        ds = DataSet(self.name, self.path)
        data = ds.preprocess(self.lemmatize, self.remove_stop, self.remove_punc, self.sent)

        vocabulary = set()
        for i, row in data.iterrows():
            vocabulary.update(row['Headline'].split(' '))
            vocabulary.update(row['Body'].split(' '))
        headlines = data.Headline.to_numpy()
        bodies = data.Body.to_numpy()

        vectorizer = TfidfVectorizer(vocabulary=vocabulary)
        headline_TF_IDF = vectorizer.fit_transform(headlines)
        body_TF_IDF = vectorizer.fit_transform(bodies)

        # Selecting the top 50 components
        svd = TruncatedSVD(n_components=50)
        headline_TF_IDF = svd.fit(headline_TF_IDF)
        body_TF_IDF = svd.fit(body_TF_IDF)

        return sklearn.metrics.pairwise.cosine_similarity(headline_TF_IDF, body_TF_IDF)

    def create_feature_file(self, path):
        features = self.get_feature()

        with open(path, 'wb') as f:
            pickle.dump(features, f)

    def read(self, name="train"):
        self.name = name
        if name == 'train':
            if not os.path.exists(SVD_FILE):
                self.create_feature_file(SVD_FILE)
            return np.array(pickle.load(open(SVD_FILE, 'rb'))).reshape(-1, 1)
        else:
            if not os.path.exists(SVD_TEST_FILE):
                self.create_feature_file(SVD_TEST_FILE)
            return np.array(pickle.load(open(SVD_TEST_FILE, 'rb'))).reshape(-1, 1)