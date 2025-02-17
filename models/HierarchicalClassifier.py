import cosine_similarity
from n_gram_matching import n_gram_matching
from sentiment import sentiment_feature
from SVD import SVD
from TFIDF import TFIDF
import baseline_features
from cue_words import cue_words
import numpy as np
from preprocessing import DataSet
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import resample


class HierarchicalClassifier:
    def __init__(self, clf1, clf2):
        self.clf1 = clf1
        self.clf2 = clf2
        features = [cosine_similarity.CosineSimilarity(),
                    n_gram_matching.NGramMatching(),
                    sentiment_feature.SentimentFeature(),
                    SVD.SVD(),
                    TFIDF.TFIDF(),
                    baseline_features.BaselineFeature(),
                    cue_words.CueWords()]
        self.features_train = np.hstack([feature.read() for feature in features])
        self.labels_train_original = DataSet(path="../FNC-1").get_labels()
        self.features_test = np.hstack([feature.read('competition_test') for feature in features])
        self.labels_test = DataSet(path="../FNC-1", name="competition_test").get_labels()
        self.features_train_new = None

    def train(self):
        # Training
        labels_train_new = [label if label == 'unrelated' else 'related' for label in self.labels_train_original]
        unrelated_indices = []
        related_indices = []
        for i, label in enumerate(labels_train_new):
            if label == 'related':
                related_indices.append(i)
            else:
                unrelated_indices.append(i)
        unrelated_indices = resample(unrelated_indices, n_samples=30000)
        related_indices = resample(related_indices, n_samples=30000)
        indices_final = related_indices + unrelated_indices
        labels_train_new = np.array(labels_train_new)[indices_final]
        features_train_new = self.features_train[indices_final]

        self.clf1.fit(features_train_new, labels_train_new)

        discuss_indices = []
        agree_indices = []
        disagree_indices = []
        for i, label in enumerate(self.labels_train_original):
            if label == 'discuss':
                discuss_indices.append(i)
            elif label == 'agree':
                agree_indices.append(i)
            elif label == 'disagree':
                disagree_indices.append(i)
        discuss_indices = resample(discuss_indices, n_samples=5000)
        agree_indices = resample(agree_indices, n_samples=5000)
        disagree_indices = resample(disagree_indices, n_samples=5000)
        indices_final = discuss_indices + agree_indices + disagree_indices
        labels_train_new = np.array(self.labels_train_original)[indices_final]
        features_train_new = self.features_train[indices_final]

        self.clf2.fit(features_train_new, labels_train_new)

    def mixed_F1(self, true, pred):
        micro = f1_score(true, pred, average='micro')
        macro = f1_score(true, pred, average='macro')
        return (macro + micro) / 2

    def evaluate(self, y_actual, y_pred):
        print("Accuracy : ", accuracy_score(y_actual, y_pred))
        print("Confusion matrix: ", confusion_matrix(y_actual, y_pred))
        print("F1 Score (macro): ", f1_score(y_actual, y_pred, average='macro'))
        print("F1 Score (micro): ", f1_score(y_actual, y_pred, average='micro'))
        print("F1 Score (weighted): ", f1_score(y_actual, y_pred, average='weighted'))
        print("F1 Score: ", f1_score(y_actual, y_pred, average=None))
        print("F1-Score (mixed): " + str(self.mixed_F1(y_actual, y_pred)))

    def predict(self, features):
        predictions = []
        for row in features:
            p = self.clf1.predict(row.reshape(1, -1))
            if p[0] == 'unrelated':
                predictions.append('unrelated')
            else:
                predictions.append(self.clf2.predict(row.reshape(1, -1))[0])
        return predictions

    def test(self):
        print("Evaluation metrics for train data:")
        self.evaluate(self.labels_train_original, self.predict(self.features_train))
        print()
        print("Evaluation metrics for test data:")
        self.evaluate(self.labels_test, self.predict(self.features_test))

    def test_clf1(self):
        predictions = self.clf1.predict(self.features_test)
        print("Evaluation metrics for test data on clf1:")
        labels_test_binary = [label if label == 'unrelated' else 'related' for label in self.labels_test]
        self.evaluate(labels_test_binary, predictions)


parameter_space = {
        'hidden_layer_sizes': [(200, 100), (100, 50), (50, 10), (10, 5)]
    }

'''
for hidden_layer_size_mlp1 in parameter_space['hidden_layer_sizes']:
    mlp1 = MLPClassifier(solver='adam',
                         activation='relu',
                         hidden_layer_sizes=hidden_layer_size_mlp1,
                         learning_rate='adaptive',
                         learning_rate_init=0.001,
                         max_iter=1000)

    print("PERFORMANCE WITH HIDDEN LAYER SIZES: " + str(hidden_layer_size_mlp1))
    hc = HierarchicalClassifier(mlp1, MLPClassifier())
    hc.train()
    hc.test_clf1()

    print("\n\n\n")
'''

# Below, we inserted (100, 50) as the hidden layer sizes for the first classifier, as this was the best found
# by the code above (commented out)
'''
for hidden_layer_size_mlp2 in parameter_space['hidden_layer_sizes']:
    mlp1 = MLPClassifier(solver='adam',
                         activation='relu',
                         hidden_layer_sizes=(100, 50),
                         learning_rate='adaptive',
                         learning_rate_init=0.001,
                         max_iter=1000)

    mlp2 = MLPClassifier(solver='adam',
                         activation='relu',
                         hidden_layer_sizes=hidden_layer_size_mlp2,
                         learning_rate='adaptive',
                         learning_rate_init=0.001,
                         max_iter=1000)

    hc = HierarchicalClassifier(mlp1, MLPClassifier())
    hc.train()
    hc.test()
'''

# Below, we inserted (100, 50) as the hidden layer sizes for the first classifier, and (50, 10) as the hidden
# layer sizes for the second classifier, as these were the best found by the code above (commented out now).
mlp1 = MLPClassifier(solver='adam',
                     activation='relu',
                     hidden_layer_sizes=(100, 50),
                     learning_rate='adaptive',
                     learning_rate_init=0.001,
                     max_iter=1000)

mlp2 = MLPClassifier(solver='adam',
                     activation='relu',
                     hidden_layer_sizes=(50, 10),
                     learning_rate='adaptive',
                     learning_rate_init=0.001,
                     max_iter=1000)

hc = HierarchicalClassifier(mlp1, mlp2)
hc.train()
hc.test()