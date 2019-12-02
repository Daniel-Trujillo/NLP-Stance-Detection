import numpy as np
from preprocessing import DataSet
import pickle
import os

CUE_WORDS_FILE = "../feature_files/cue_words_features.pkl"
CUE_WORDS_TEST_FILE = "../feature_files/cue_words_test_features.pkl"

class CueWords:
    def __init__(self, path="./FNC-1/", name="train", lemmatize=True, remove_stop=True, remove_punc=True):
        self.path = path
        self.name = name
        self.lemmatize = lemmatize
        self.remove_stop = remove_stop
        self.remove_punc = remove_punc

    def get_cue_words(self):
        words_list = []
        with open("./cue_words/cue_word_list.txt", "r") as f:
            for line in f.readlines():
                words_list.append(line.strip().lower())
        return words_list

    def get_feature(self):
        ds = DataSet(path=self.path, name=self.name)
        data = ds.preprocess(self.lemmatize, self.remove_stop, self.remove_punc)
        cue_words_list = self.get_cue_words()
        X = []
        for index, row in data.iterrows():
            X_row = []
            for word in cue_words_list:
                if word in row['Headline']:
                    X_row.append(1)
                else:
                    X_row.append(0)
            for word in cue_words_list:
                if word in row['Body']:
                    X_row.append(1)
                else:
                    X_row.append(0)
            X.append(X_row)
        return np.array(X)

    def create_feature_file(self, path):
        features = self.get_feature()

        with open(path, 'wb') as f:
            pickle.dump(features, f)

    def read(self, name="train"):
        self.name = name
        if name == 'train':
            if not os.path.exists(CUE_WORDS_FILE):
                self.create_feature_file(CUE_WORDS_FILE)
            return np.array(pickle.load(open(CUE_WORDS_FILE, 'rb')))
        else:
            if not os.path.exists(CUE_WORDS_TEST_FILE):
                self.create_feature_file(CUE_WORDS_TEST_FILE)
            return np.array(pickle.load(open(CUE_WORDS_TEST_FILE, 'rb')))