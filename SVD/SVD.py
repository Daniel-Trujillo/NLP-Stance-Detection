from gensim.models import KeyedVectors
import sklearn
import pickle
from preprocessing import DataSet
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

COSINE_SIMILARITY_FILE = "../feature_files/similarity_features.pkl"
COSINE_SIMILARITY_TEST_FILE = "../feature_files/similarity_test_features.pkl"


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
        #
        # vocabulary = [word for line in list(np.concatenate((headlines, bodies), axis=None)) for word in line]
        #
        # # Getting rid of duplicates
        # vocabulary = set(vocabulary)

        # Making headlines and bodies strings
        # headlines = [" ".join(line) for line in headlines]
        # bodies = [" ".join(line) for line in bodies]

        vectorizer = TfidfVectorizer(vocabulary=vocabulary)
        headline_TF_IDF = vectorizer.fit_transform(headlines)
        body_TF_IDF = vectorizer.fit_transform(bodies)

        return sklearn.metrics.pairwise.cosine_similarity(headline_TF_IDF, body_TF_IDF)

    def create_feature_file(self, path):
        features = self.get_feature()

        with open(path, 'wb') as f:
            pickle.dump(features, f)

    def read(self, name="train"):
        self.name = name
        if name == 'train':
            if not os.path.exists(COSINE_SIMILARITY_FILE):
                self.create_feature_file(COSINE_SIMILARITY_FILE)
            return np.array(pickle.load(open(COSINE_SIMILARITY_FILE, 'rb'))).reshape(-1, 1)
        else:
            if not os.path.exists(COSINE_SIMILARITY_TEST_FILE):
                self.create_feature_file(COSINE_SIMILARITY_TEST_FILE)
            return np.array(pickle.load(open(COSINE_SIMILARITY_TEST_FILE, 'rb'))).reshape(-1, 1)



sd = SVD()
test = sd.get_feature()
print(len(test))