from sklearn.neural_network import MLPClassifier
from models import Models
from sklearn.metrics import f1_score, make_scorer

def mixed_F1(true, pred):
    micro = f1_score(true, pred, average='micro')
    macro = f1_score(true, pred, average='macro')

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
best_params, mean, std = models.grid_search(parameters=parameter_space, k=10, scoring=make_scorer(mixed_F1))

print(mean)
print(std)

print(str(best_params))


lr = MLPClassifier(solver=best_params['solver'], activation=best_params['activation'],
                   hidden_layer_sizes=best_params['hidden_layer_sizes'], learning_rate=best_params['learning_rate'],
                   max_iter=10000)

models.model = lr
models.train()

predictions, true = models.test()

F1_score = mixed_F1(true, predictions)
print("F1-Score testing: " + str(F1_score))
