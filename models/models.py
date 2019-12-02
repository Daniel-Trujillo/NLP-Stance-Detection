from average_word_embeddings import cosine_similarity
from n_gram_matching import n_gram_matching
from sentiment import sentiment_feature
from SVD import SVD
from TFIDF import TFIDF
import baseline_features
from cue_words import cue_words
import numpy as np
from preprocessing import DataSet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

class Models:
    def __init__(self, modelInstance):
        self.model = modelInstance
        features = [cosine_similarity.CosineSimilarity(),
                    n_gram_matching.NGramMatching(),
                    sentiment_feature.SentimentFeature(),
                    SVD.SVD(),
                    TFIDF.TFIDF(),
                    baseline_features.BaselineFeature(),
                    cue_words.CueWords()]
        self.features_train = np.hstack([feature.read() for feature in features])
        self.labels_train = DataSet(path="../FNC-1").get_labels()
        self.features_test = np.hstack([feature.read('competition_test') for feature in features])
        self.labels_test = DataSet(path="../FNC-1", name="competition_test").get_labels()

    def train(self):
        self.model.fit(self.features_train, self.labels_train)
        return self.model

    def mixed_F1(self, true, pred):
        micro = f1_score(true, pred, average='micro')
        macro = f1_score(true, pred, average='macro')
        return (macro + micro) / 2

    def evaluate(self, y_actual, y_pred):
        print("Accuracy : ", accuracy_score(y_actual, y_pred))
        print("Confusion matrix: \n", confusion_matrix(y_actual, y_pred))
        print("F1 Score (macro): ", f1_score(y_actual, y_pred, average='macro'))
        print("F1 Score (micro): ", f1_score(y_actual, y_pred, average='micro'))
        print("F1 Score (weighted): ", f1_score(y_actual, y_pred, average='weighted'))
        print("F1 Score: ", f1_score(y_actual, y_pred, average=None))
        print("F1-Score (mixed): " + str(self.mixed_F1(y_actual, y_pred)))

    def test(self):
        y_pred = self.model.predict(self.features_train)
        print("Evaluation metrics for train data:")
        self.evaluate(self.labels_train, y_pred)
        print()
        print("Evaluation metrics for test data:")
        y_pred = self.model.predict(self.features_test)
        self.evaluate(self.labels_test, y_pred)
        return self.model.predict(self.features_test), self.labels_test

    def grid_search(self, parameters, k, scoring):
        clf_cv = GridSearchCV(self.model, param_grid=parameters, cv=k, scoring=scoring)
        clf_cv.fit(self.features_train, self.labels_train)
        mean = clf_cv.cv_results_['mean_test_score'][0]
        std = clf_cv.cv_results_['std_test_score'][0]
        return clf_cv.best_params_, mean, std
