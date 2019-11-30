from gensim.models import KeyedVectors
import sklearn
import pickle
from preprocessing import DataSet
import os
import numpy as np

# nltk.download('wordnet')
# nltk.download('stopwords')

COSINE_SIMILARITY_FILE = "../feature_files/similarity_features.pkl"
COSINE_SIMILARITY_TEST_FILE = "../feature_files/similarity_test_features.pkl"


class CosineSimilarity:
    def __init__(self, name="train", path="../FNC-1", lemmatize=True, remove_stop=True, remove_punc=False):
        self.model = KeyedVectors.load_word2vec_format('../average_word_embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
        self.bodies = {}
        self.path = path
        self.name = name
        self.lemmatize = lemmatize
        self.remove_stop = remove_stop
        self.remove_punc = remove_punc

    def get_feature(self):
        ds = DataSet(self.name, self.path)
        data = ds.preprocess(self.lemmatize, self.remove_stop, self.remove_punc)
        cosine_similarities = []

        for index, row in data.iterrows():
            headline = row['Headline']
            body = row['Body']
            bodyID = row['BodyID']

            word_embeddings_headline = [self.model[word] for word in headline if word in self.model.vocab]
            average_headline = [sum(column) / len(column) for column in zip(*word_embeddings_headline)]

            # As multiple stances use the same bodies, we store them in a dictionary
            if bodyID not in self.bodies:
                word_embeddings_body = [self.model[word] for word in body if word in self.model.vocab]
                average_body = [sum(column) / len(column) for column in zip(*word_embeddings_body)]
                self.bodies[bodyID] = average_body

            average_body = self.bodies[bodyID]

            if len(average_headline) == 0 or len(average_body) == 0:
                cosine_similarities.append(0)
            else:
                cosine_similarities.append(sklearn.metrics.pairwise.cosine_similarity([average_headline], [average_body])[0][0])

        return cosine_similarities

    def create_feature_file(self, path):
        features = self.get_feature()

        with open(path, 'wb') as f:
            pickle.dump(features, f)

    def read(self, name="train"):
        self.name = name
        if type == 'train':
            if not os.path.exists(COSINE_SIMILARITY_FILE):
                self.create_feature_file(COSINE_SIMILARITY_FILE)
            return np.array(pickle.load(open(COSINE_SIMILARITY_FILE, 'rb'))).reshape(-1, 1)
        else:
            if not os.path.exists(COSINE_SIMILARITY_TEST_FILE):
                self.create_feature_file(COSINE_SIMILARITY_TEST_FILE)
            return np.array(pickle.load(open(COSINE_SIMILARITY_TEST_FILE, 'rb'))).reshape(-1, 1)

