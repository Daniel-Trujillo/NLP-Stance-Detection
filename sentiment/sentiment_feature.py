from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from preprocessing import DataSet
import pickle
import os

import nltk
nltk.download('vader_lexicon')

SENTIMENT_FILE = "../feature_files/sentiment_features.pkl"
SENTIMENT_TEST_FILE = "../feature_files/sentiment_test_features.pkl"


class SentimentFeature:
    def __init__(self, path="../FNC-1/", name="train", lemmatize=True, remove_stop=True, remove_punc=False, sent=True):
        self.path = path
        self.name = name
        self.lemmatize = lemmatize
        self.remove_stop = remove_stop
        self.remove_punc = remove_punc
        self.sent = sent

    def get_feature(self):
        def get_sentiment(d):
            return list(sid.polarity_scores(d).values())
        ds = DataSet(path=self.path, name=self.name)
        data = ds.preprocess(self.lemmatize, self.remove_stop, self.remove_punc, self.sent)
        sid = SentimentIntensityAnalyzer()
        sentiments = []
        for index, row in data.iterrows():
            headline_sentiment = get_sentiment(row['Headline'])
            body_sentiment = get_sentiment(row['Body'])
            sentiments.append([headline_sentiment + body_sentiment])
        return np.array(sentiments).reshape(-1, 8)

    def create_feature_file(self, path):
        features = self.get_feature()

        with open(path, 'wb') as f:
            pickle.dump(features, f)

    def read(self, name="train"):
        self.name = name
        if type == 'train':
            if not os.path.exists(SENTIMENT_FILE):
                self.create_feature_file(SENTIMENT_FILE)
            return np.array(pickle.load(open(SENTIMENT_FILE, 'rb'))).reshape(-1, 1)
        else:
            if not os.path.exists(SENTIMENT_TEST_FILE):
                self.create_feature_file(SENTIMENT_TEST_FILE)
            return np.array(pickle.load(open(SENTIMENT_TEST_FILE, 'rb'))).reshape(-1, 1)