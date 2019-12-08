from models import Models
from sklearn.metrics import f1_score, make_scorer
from sklearn.svm import SVC

def mixed_F1(true, pred):
    micro = f1_score(true, pred, average='micro')
    macro = f1_score(true, pred, average='macro')

    return (macro + micro) / 2


parameter_space = {
        'kernel': ['rbf', 'sigmoid'],
        'C': [1, 10, 100],
        'gamma': [0, 5, 10, 'auto']
    }

clf = SVC()
models = Models(clf)
best_params, mean, std = models.grid_search(parameters=parameter_space, k=10, scoring=make_scorer(mixed_F1))

print("Best Params:")
print(best_params)
print("mean")
print(mean)
print("std")
print(std)

clf = SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'], max_iter=10000)

models.model = clf
models.train()

predictions, true = models.test()

F1_score = mixed_F1(true, predictions)
print("F1-Score testing: " + str(F1_score))