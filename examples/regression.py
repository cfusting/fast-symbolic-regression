import numpy as np

from fastsr.estimators.symbolic_regression import SymbolicRegression


def target(x):
    return x**3 + x**2 + x

X = np.linspace(-10, 10, 100, endpoint=True)
y = target(X)

sr = SymbolicRegression()
sr.fit(X, y)
score = sr.score(X, y)
print('Score: ' + str(score))
print('Best Individuals:')
sr.print_best_individuals()
