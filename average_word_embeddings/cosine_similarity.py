from preprocessing import DataSet
from gensim.models import KeyedVectors
import sklearn
import pickle
from preprocessing import DataSet

#nltk.download('wordnet')
#nltk.download('stopwords')

class CosineSimilarity():
    def __init__(self, name="train", path="../FNC-1", lemmatize=True, remove_stop=True, remove_punc=False):
        self.model = KeyedVectors.load_word2vec_format('../average_word_embeddings/GoogleNews-vectors-negative300.bin', binary=True)
        self.bodies = {}
        self.ds = DataSet(name, path)
        self.data = self.ds.preprocess(lemmatize, remove_stop, remove_punc)

    def get_feature(self, data):
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
        features = self.get_feature(self.data)

        with open(path, 'wb') as f:
            pickle.dump(features, f)

