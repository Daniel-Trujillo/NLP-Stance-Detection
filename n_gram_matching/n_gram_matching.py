import numpy as np
from preprocessing import DataSet
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

N_GRAM_FEATURE_FILE = "../feature_files/ngram_matching.pkl"
N_GRAM_FEATURE_TEST_FILE = "../feature_files/ngram_matching_test.pkl"


def tokenizer(x):
    return x

def preprocessor(x):
    return x

class NGramMatching:
    def __init__(self, path="../FNC-1/", name="train", lemmatize=True, remove_stop=True, remove_punc=False):
        self.path = path
        self.name = name
        self.lemmatize = lemmatize
        self.remove_stop = remove_stop
        self.remove_punc = remove_punc

    def get_ngram(self, n, text):
        ngrams = {}
        for i in range(n, len(text) + 1):
            ngram_words = tuple(text[i - n:i])
            if ngram_words in ngrams.keys():
                ngrams[ngram_words] += 1
            else:
                ngrams[ngram_words] = 1
        return ngrams

    def getIDF(self, tokens):
        if self.name == 'train':
            vectorizer = TfidfVectorizer(
                use_idf=True,
                tokenizer=tokenizer,
                preprocessor=preprocessor,
                ngram_range=(1, 5)
            )
            vectorizer.fit(tokens)
            with open("../feature_files/ngram_tfidf_vectorizer.pkl", 'wb') as f:
                pickle.dump(vectorizer, f)
            vectorizer.transform(tokens)
        else:
            vectorizer = pickle.load(open("../feature_files/ngram_tfidf_vectorizer.pkl", 'rb'))
        idf = vectorizer.idf_
        return dict(zip(vectorizer.get_feature_names(), idf))

    def nGramMathing(self):
        ds = DataSet(path=self.path, name=self.name)
        data = ds.preprocess(self.lemmatize, self.remove_stop, self.remove_punc)
        idf = self.getIDF(data["Body"].to_numpy())
        features = []
        for index, row in data.iterrows():
            H = []
            A = []
            for n in range(1, 6):
                H_ngram = self.get_ngram(n, row['Headline'])
                A_ngram = self.get_ngram(n, row["Body"])
                H.extend(list(H_ngram.keys()))
                A.extend(list(A_ngram.keys()))
            sum = 0
            for i, h in enumerate(H):
                TF_hi = (H.count(h) + A.count(h)) * len(h)
                idf_hi = idf.get(" ".join(h), 0)
                sum += (TF_hi * idf_hi)
            sc = sum / (len(H) + len(A))
            features.append(sc)
        return np.array(features).reshape(-1, 1)

    def create_feature_file(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.nGramMathing(), f)

    def read(self, name="train"):
        self.name = name
        if name == 'train':
            if not os.path.exists(N_GRAM_FEATURE_FILE):
                self.create_feature_file(N_GRAM_FEATURE_FILE)
            return np.array(pickle.load(open(N_GRAM_FEATURE_FILE, 'rb'))).reshape(-1, 1)
        else:
            if not os.path.exists(N_GRAM_FEATURE_TEST_FILE):
                self.create_feature_file(N_GRAM_FEATURE_TEST_FILE)
            return np.array(pickle.load(open(N_GRAM_FEATURE_TEST_FILE, 'rb'))).reshape(-1, 1)
