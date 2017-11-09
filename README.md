# Blazing Fast Symbolic Regression

fastsr is a symbolic regression package built on top of fastgp, a numpy implementation of genetic programming built on top of deap.

Example Usage
-------------
Symbolic Regression is really good at fitting nonlinear functions. Let's try to fit the third order polynomial x^3 + x^2 + x.
```python
def target(x):
    return x**3 + x**2 + x
```
Now we'll generate some data on the domain \[-10, 10\).
```python
X = np.arange(-10, 10, .25)
X = X.reshape((len(X), 1))
y = np.apply_along_axis(target, 0, X).reshape(len(X))
```
Finally we'll create and fit the Symbolic Regression estimator and check the score.
```python
sr = SymbolicRegression()
sr.fit(X, y)
score = sr.score(X, y)
```
```
Score: 0.0
```
Whoa! That's not much error. Don't get too used to scores like that though, real data sets aren't usually as simple as a third order polynomial.

fastsr uses Genetic Programming to fit the data. That means equations are evolving to fit the data better and better each generation. Let's have a look at the best individuals and their respective scores.
```pyton
print('Best Individuals:')
sr.print_best_individuals()
```
```
Best Individuals:
0.0 : add(add(square(X0), X0), cube(X0))
33.34375 : add(square(X0), cube(X0))
2000.127354695976 : add(cbrt(X0), cbrt(cube(cube(X0))))
2002.083203125 : add(X0, cbrt(cube(cube(X0))))
2002.083203125 : add(cube(X0), X0)
2010.426953125 : cube(X0)
71748.56942173702 : exp(multiply(numpy_protected_log_abs(X0), numpy_protected_sqrt(X0)))
134846.70314941407 : add(X0, add(X0, X0))
138725.83830566407 : add(X0, X0)
138725.83830566407 : add(X0, cbrt(cube(X0)))
140572.96838378906 : add(square(numpy_protected_sqrt(X0)), numpy_protected_log_abs(exp(X0)))
142671.66096191405 : X0
145678.76900113834 : cbrt(X0)
146593.42518662894 : numpy_protected_sqrt(numpy_protected_log_abs(cube(X0)))
8.648831663174305e+40 : square(exp(cube(numpy_protected_sqrt(exp(cbrt(X0))))))
```
At the top we find our best individual, which is exactly the third order polynomial we defined our target function to be. You might be confused as to why we consider all these other individuals, some with very large errors be be "best".