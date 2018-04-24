from sklearn.neibhbors import NearestNeghbors
import numpy as np


def tune(X_train, y_train, scoring):

    # penalty: Regularization Term -> 'l2' or 'l1' or 'elasticnet'
    penalty = [None, 'l2', 'l1', 'elasticnet']
    # alpha: (float) constant that multiplies regulatrization term
    alpha = [float(10.0**x) for x in np.linspace(-3, 3, num = 6+1)]
    # max_iter: (int)
    max_iter = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Create the random grid
    random_grid = {'penalty': penalty,
                    'alpha': alpha,
                    'max_iter': max_iter}

    fracNeg = len(y_train[y_train == 0])/float(len(y_train))
    weight = (1-fracNeg)/float(fracNeg) 
    class_weight = {1:1, 0:weight}

    perceptron = Perceptron(class_weight=class_weight)

    perceptron_random = \
        RandomizedSearchCV(estimator = perceptron, param_distributions = random_grid, \
                            n_iter = 100, cv = 3, verbose=2, random_state=42, \
                            n_jobs = -1, scoring=scoring)
    
    perceptron_random.fit(X_train, y_train)
    return perceptron_random.best_params_

def main():
    pass

if __name__ == '__main__':
    main()