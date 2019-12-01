
import numpy as np
import gensim.downloader as api
from preprocessing import DataSet
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class CueWords:
    def __init__(self, path="../FNC-1/", name="train", lemmatize=True, remove_stop=True, remove_punc=False):
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
        wv = api.load('word2vec-google-news-300')
        sims = []
        for word in words_list:
            try:
                similars = wv.most_similar(word, topn=5)
                for sim in similars:
                    sims.append(sim[0])
            except KeyError:
                pass
        words_list.extend(sims)
        return words_list


    def cue_words(self):
        ds = DataSet(path=self.path, name=self.name)
        data = ds.preprocess(self.lemmatize, self.remove_stop, self.remove_punc)
        


cw = CueWords()
