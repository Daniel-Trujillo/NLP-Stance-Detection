import sklearn
import pickle
from preprocessing import DataSet
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

TFIDF_FILE = "../feature_files/tfidf_features.pkl"
TFIDF_TEST_FILE = "../feature_files/tfidf_test_features.pkl"


class TFIDF:
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
        headline_body = []
        for i, row in data.iterrows():
            vocabulary.update(row['Headline'].split(' '))
            vocabulary.update(row['Body'].split(' '))
            headline_body.append(row['Headline'] + ' ' + row['Body'])
        headlines = data.Headline.to_numpy()
        bodies = data.Body.to_numpy()

        if self.name == 'train':
            vectorizer = TfidfVectorizer(vocabulary=vocabulary)
            headline_body_TF_IDF = vectorizer.fit(headline_body)
            with open('../feature_files/headline_body_tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(headline_body_TF_IDF, f)
            headline_TF_IDF = headline_body_TF_IDF.transform(headlines)
            body_TF_IDF = headline_body_TF_IDF.transform(bodies)
        else:
            headline_body_TF_IDF = pickle.load(open('../feature_files/headline_body_tfidf_vectorizer.pkl', 'rb'))
            headline_TF_IDF = headline_body_TF_IDF.transform(headlines)
            body_TF_IDF = headline_body_TF_IDF.transform(bodies)

        features = []
        for h, b in zip(headline_TF_IDF, body_TF_IDF):
            features.append(sklearn.metrics.pairwise.cosine_similarity(h,b)[0][0])
        return np.array(features)

    def create_feature_file(self, path):
        features = self.get_feature()

        with open(path, 'wb') as f:
            pickle.dump(features, f)

    def read(self, name="train"):
        self.name = name
        if name == 'train':
            if not os.path.exists(TFIDF_FILE):
                self.create_feature_file(TFIDF_FILE)
            return np.array(pickle.load(open(TFIDF_FILE, 'rb'))).reshape(-1, 1)
        else:
            if not os.path.exists(TFIDF_TEST_FILE):
                self.create_feature_file(TFIDF_TEST_FILE)
            return np.array(pickle.load(open(TFIDF_TEST_FILE, 'rb'))).reshape(-1, 1)