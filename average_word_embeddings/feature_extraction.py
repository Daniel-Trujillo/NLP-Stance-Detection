from preprocessing import DataSet
from gensim.models import KeyedVectors
import sklearn
import pickle

#nltk.download('wordnet')
#nltk.download('stopwords')

class CosineSimilarity():
    def __init__(self):
        # load the google word2vec model
        self.model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        self.bodies = {}

    def get_features(self, data):
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

            cosine_similarities.append(sklearn.metrics.pairwise.cosine_similarity([average_headline], [average_body])[0][0])

        return cosine_similarities

    def create_features_file(self, data, path):
        features = self.get_features(data)

        with open(path, 'wb') as f:
            pickle.dump(features, f)


ds = DataSet(path='../FNC-1')
df = ds.preprocess(lemmatize=True, remove_stop=True, remove_punc=False)

cs = CosineSimilarity()
print(cs.get_features(df))

