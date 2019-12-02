
import numpy as np
from preprocessing import DataSet
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class CueWords:
    def __init__(self, path="./FNC-1/", name="train", lemmatize=True, remove_stop=True, remove_punc=False):
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


    def cue_words(self):
        ds = DataSet(path=self.path, name=self.name)
        data = ds.preprocess(self.lemmatize, self.remove_stop, self.remove_punc)
        cue_words_list = self.get_cue_words()
        X = []
        Y = []
        for index, row in data.iterrows():
            Y.append(row['Stance'])
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


cw = CueWords()
