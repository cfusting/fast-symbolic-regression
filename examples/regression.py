import matplotlib.pyplot as plt

import numpy as np

from fastsr.estimators.symbolic_regression import SymbolicRegression

from fastgp.algorithms.fast_evaluate import fast_numpy_evaluate
from fastgp.parametrized.simple_parametrized_terminals import get_node_semantics


def target(x):
    return x**3 + x**2 + x

X = np.linspace(-10, 10, 100, endpoint=True)
y = target(X)

seed = 72066
sr = SymbolicRegression(seed=seed)
sr.fit(X, y)
score = sr.score(X, y)
print('Score: ' + str(score))
print('Best Individuals:')
sr.print_best_individuals()

history = sr.history_
population = list(filter(lambda x: hasattr(x, 'error'), list(sr.history_.genealogy_history.values())))
population.sort(key=lambda x: x.error, reverse=True)

X = X.reshape((len(X), 1))
i = 1
previous_errror = population[0]
unique_individuals = []
while i < len(population):
    ind = population[i]
    if ind.error != previous_errror:
        print(str(i) + ' | ' + str(ind.error) + ' | ' + str(ind))
        unique_individuals.append(ind)
    previous_errror = ind.error
    i += 1


def plot(index):
    plt.plot(X, y, 'r')
    plt.axis([-10, 10, -1000, 1000])
    y_hat = fast_numpy_evaluate(unique_individuals[index], sr.pset_.context, X, get_node_semantics)
    plt.plot(X, y_hat, 'g')
    plt.savefig(str(i) + 'ind.png')
    plt.gcf().clear()

i = 0
while i < len(unique_individuals):
    plot(i)
    i += 10
i = len(unique_individuals) - 1
plot(i)
