import numpy as np

from fastsr.estimators.symbolic_regression import SymbolicRegression


def target(x):
    return x**3 + x**2 + x

X = np.arange(-10, 10, .25)
X = X.reshape((len(X), 1))
y = np.apply_along_axis(target, 0, X).reshape(len(X))

sr = SymbolicRegression()
sr.fit(X, y)
score = sr.score(X, y)
print('Score: ' + str(score))
print('Best Individuals:')
sr.print_best_individuals()
