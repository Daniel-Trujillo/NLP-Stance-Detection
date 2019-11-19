from sklearn.linear_model import LogisticRegression
from models import Models
from sklearn.metrics import f1_score, make_scorer, confusion_matrix
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
        'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
        'fit_intercept': [True, False],
        'multi_class': ['ovr', 'multinomial', 'auto'],
        'max_iter': [1000],
    }

# Logistic Regression
lr = LogisticRegression()

models = Models(lr)

best_params, mean, std = models.grid_search(model=lr, parameters=parameter_space, k=10, scoring=make_scorer(own_F1_score))

print(best_params)

print("Best parameters found a mean F1-score of " + str(mean) + " with a standard deviation of " + str(std))

lr = LogisticRegression(fit_intercept=best_params['fit_intercept'], max_iter=10000, multi_class=best_params['multi_class'], solver=best_params['solver'])
models.model = lr

models.train()

predictions, true = models.test()

F1_score = own_F1_score(true, predictions)
print("F1-Score testing: " + str(F1_score))
