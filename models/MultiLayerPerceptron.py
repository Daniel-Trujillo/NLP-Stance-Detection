from sklearn.neural_network import MLPClassifier
from models import Models
from sklearn.metrics import make_scorer, confusion_matrix
import numpy as np

def own_F1_score(true, pred):
    confusion = confusion_matrix(true, pred)
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - TP
    FN = confusion.sum(axis=1) - TP

    F1 = (2 * TP) / (2 * TP + FP + FN)

    macro = np.sum(F1) / F1.size
    micro = 2 * np.sum(TP) / (2 * np.sum(TP) + np.sum(FP) + np.sum(FN))

    return (macro + micro) / 2

parameter_space = {
        'solver': ['sgd', 'adam'],
        'activation': ['tanh', 'relu'],
        'hidden_layer_sizes': [(10, 10), (10, 100)],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [1000],
    }

lr = MLPClassifier()
models = Models(lr)
best_params, mean, std = models.grid_search(model=lr, parameters=parameter_space, k=10, scoring=make_scorer(own_F1_score))

print(mean)
print(std)

print(str(best_params))

lr = MLPClassifier(solver=best_params['solver'], activation=best_params['activation'],
                   hidden_layer_sizes=best_params['hidden_layer_sizes'], learning_rate=best_params['learning_rate'],
                   max_iter=10000)

models.model = lr
models.train()

predictions, true = models.test()

F1_score = own_F1_score(true, predictions)
print("F1-Score testing: " + str(F1_score))
