from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from models import Models


# Logistic Regression
lr = LogisticRegression()
models = Models(lr)
lr = models.train()
predictions = models.test()

#
mlp = MLPClassifier()
models = Models(mlp)
models = models.train()
predictions = models.test()

