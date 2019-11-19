from sklearn.model_selection import GridSearchCV

def grid_search(model, x, y, parameters, k, scoring):
    clf = GridSearchCV(model, parameters, cv=k, scoring=scoring)

    clf.fit(x. y)

    mean = clf.cv_results_['mean_test_score'][0]
    std = clf.cv_results_['std_test_score'][0]

    return clf.best_params, mean, std
