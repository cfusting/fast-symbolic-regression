import random
import pickle

from sklearn.base import BaseEstimator

import fastsr.containers.learning_data as ld
from fastsr.experiments.control import Control

import numpy as np


def check_data(dat):
    if dat.dtype == object:
        dat = dat.astype(float)
    if dat.shape[0] == 0:
        raise ValueError('Bad data: Zero observations.')
    if len(dat.shape) > 1 and dat.shape[1] == 0:
        raise ValueError('Bad data: Zero variables.')
    if not np.all(np.isfinite(dat)):
        raise ValueError('Bad data: Non finite values.')


def reshape_y(why):
    if len(why.shape) > 1:
        return why.reshape(len(why))
    return why


def reshape_X(ex):
    if len(ex.shape) < 2:
        return ex.reshape((len(ex), 1))
    return ex


def reshape_dat(ex, why):
    return reshape_X(ex), reshape_y(why)


def init_dat(ex, why):
    check_data(ex)
    check_data(why)
    ex, why = reshape_dat(ex, why)
    return ex, why


class SymbolicRegression(BaseEstimator):
    """A Scikit-Learn compatible wrapper for Symbolic Regression.

    Parameters
    ----------
    experiment_class : class
        Class of the experiment containing the underlying implementation
        of Symbolic Regression.

    variable_type_indices : list
        An list of tuples delineating the (inclusive, exclusive) boundaries
        of the variable types in X. Necessary for the use of RangeTerimals.

    variable_names : list
        Array of variable names.

    variable_dict : dict
        Dictionary of default Deap variable names to variable_names.

    ngen : integer
        Number of generations to evolve individuals.

    pop_size : integer
        Number of individuals to evolve.

    tournament_size : integer
        Number of individuals participating in a fitness tournament.

    min_depth_init : integer
        Minimum depth of randomly initialized individuals.

    max_dept_init : integer
        Maximum depth of randomly initialized individuals.

    max_height : integer
        Maximum height an individual can become.

    max_size : integer
        Maximum number of nodes an individual can be made up of.

    crossover_probability : float
        Probability that individuals will mate.

    mutation_probability : float
        Probability that individuals will mutate.

    internal_node_selection_bias : float
        Probability an internal node will be selected during mating and mutation.

    min_gen_grow : integer
        Minimum height of trees grown for mutations.

    max_gen_grow : integer
        Maximum height of trees grown for mutations.

    subset_proportion : float
        Proportion of data to be used for training.

    subset_change_frequency : integer
        Number of generations before data subset sample is changed.

    num_randoms : integer
        Number of randomly generated individuals to insert into the
        population each generation.

    seed : integer
        Random seed.

    ensemble_size : integer
        Number of lowest error individuals from best_individuals_ to use
        when making predictions and scoring the model.

    Attributes
    ----------
    population_ : list
        Individuals from the last population

    logbook_ : A Deap Logbook object
        Statistics from fitting procedure.

    history_ : A Deap History object
        Genealogy from the fitting procedure.

    best_individuals_ : list
        The best individuals from the fitting procedure.

    pset_ : A Deap PrimitiveSet object.
        The building blocks of the individuals.

    experiment_ : An implementation of an Experiment object.
        The experiment containing the underlying implementation
        of Symbolic Regression.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from fastsr.estimators.symbolic_regression import SymbolicRegression
    >>>
    >>> def target(x):
    >>>     return x**3 + x**2 + x

    >>> X = np.linspace(-10, 10, 100, endpoint=True)
    >>> y = target(X)
    >>> sr = SymbolicRegression(seed=72066)
    >>> sr.fit(X, y)
    >>> score = sr.score(X, y)
    >>> print('Score: ' + str(score))
    >>> print('Best Individuals:')
    >>> sr.print_best_individuals()
    """

    def __init__(self,
                 experiment_class=None,
                 variable_type_indices=None,
                 variable_names=None,
                 variable_dict=None,
                 num_features=None,
                 ngen=20,
                 pop_size=50,
                 tournament_size=2,
                 min_depth_init=1,
                 max_dept_init=6,
                 max_height=17,
                 max_size=200,
                 crossover_probability=0.9,
                 mutation_probability=0.1,
                 internal_node_selection_bias=0.9,
                 min_gen_grow=1,
                 max_gen_grow=6,
                 subset_proportion=1,
                 subset_change_frequency=10,
                 num_randoms=1,
                 seed=np.random.randint(10**6),
                 ensemble_size=1):

        self.experiment_class = experiment_class
        self.num_features = num_features
        self.variable_type_indices = variable_type_indices
        self.variable_names = variable_names
        self.variable_dict = variable_dict
        self.ngen = ngen
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.min_depth_init = min_depth_init
        self.max_dept_init = max_dept_init
        self.max_height = max_height
        self.max_size = max_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.internal_node_selection_bias = internal_node_selection_bias
        self.min_gen_grow = min_gen_grow
        self.max_gen_grow = max_gen_grow
        self.subset_proportion = subset_proportion
        self.subset_change_frequency = subset_change_frequency
        self.num_randoms = num_randoms
        self.ensemble_size = ensemble_size
        self.seed = seed

    def initialize_defaults(self, X):
        self.num_features = X.shape[1]
        self.experiment_class = Control
        self.variable_type_indices = [self.num_features - 1]
        self.variable_names = ['X' + str(x) for x in range(self.num_features)]
        self.variable_dict = ld.get_variable_dict(self.variable_names, ld.LearningData.DEFAULT_PREFIX)

    def fit(self, X, y):
        """
        Fit model as described in toolbox.run() in the implementation
        of Experiment.

        Parameters
        -----------
        X : ndarray (n_samples, n_features)
            Data

        y : ndarray (n_samples,)
            Target
        """

        X, y = init_dat(X, y)
        if self.experiment_class is None:
            self.initialize_defaults(X)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.experiment_ = self.experiment_class(ngen=self.ngen,
                                                 pop_size=self.pop_size,
                                                 tournament_size=self.tournament_size,
                                                 min_depth_init=self.min_depth_init,
                                                 max_dept_init=self.max_dept_init,
                                                 max_height=self.max_height,
                                                 max_size=self.max_size,
                                                 crossover_probability=self.crossover_probability,
                                                 mutation_probability=self.mutation_probability,
                                                 internal_node_selection_bias=self.internal_node_selection_bias,
                                                 min_gen_grow=self.min_gen_grow,
                                                 max_gen_grow=self.max_gen_grow,
                                                 subset_proportion=self.subset_proportion,
                                                 subset_change_frequency=self.subset_change_frequency,
                                                 num_randoms=self.num_randoms)
        if not hasattr(self, 'pset_'):
            self.pset_ = self.experiment_.get_pset(X.shape[1], self.variable_type_indices, self.variable_names,
                                                   self.variable_dict)
        toolbox = self.experiment_.get_toolbox(X, y, self.pset_, self.variable_type_indices,
                                               self.variable_names)
        self.population_, self.logbook_, self.history_ = toolbox.run()
        self.best_individuals_ = self.experiment_.get_best_individuals(self.population_)
        self.best_individuals_.sort(key=lambda x: x.error)
        if self.ensemble_size > len(self.best_individuals_):
            print('Warning: enemble_size (' + str(self.ensemble_size) + ') larger than the length '
                  'of best_individuals (' + str(len(self.best_individuals_)) + '). Lowering ensemble'
                  'size to match the length of best_individuals.')
            self.ensemble_size = len(self.best_individuals_)
        return self

    def predict(self, X):
        """Make predictions given predictor matrix X using the
        prediction_toolbox defined in the implementation of Experiment.

        Parameters
        -----------
        X : ndarray (n_samples, n_features)
            Data

        Notes
        -----
        Uses the average predictions of the lowest error ensemble_size
        individuals to make predictions.
        """
        check_data(X)
        X = reshape_X(X)
        if self.num_features != X.shape[1]:
            raise ValueError('Cannot make predictions with a different number of features than with what the model was '
                             'fit.')
        prediction_toolbox = self.experiment_.get_prediction_toolbox(X, self.pset_)
        predictions = np.zeros((X.shape[0], self.ensemble_size))
        for i in range(self.ensemble_size):
            prediction = prediction_toolbox.predict(self.best_individuals_[i])
            predictions[:, i] = prediction
        return predictions.mean(axis=1)

    def score(self, X, y):
        """Score the model using the scoring_toolbox defined in the
        Implementation of Experiment.

        Parameters
        -----------
        X : ndarray (n_samples, n_features)
            Data

        y : ndarray (n_samples,)
            Target
        """

        X, y = init_dat(X, y)
        if self.num_features != X.shape[1]:
            raise ValueError('Cannot score model with a different number of features than with what the model was '
                             'fit.')
        scoring_toolbox = self.experiment_.get_scoring_toolbox(X, y, self.pset_)
        scores = np.zeros(self.ensemble_size)
        for i in range(self.ensemble_size):
            scores[i] = scoring_toolbox.score(self.best_individuals_[i])[0]
        return scores.mean()

    def save(self, filename):
        """Save this model for later use.
        """

        # Need to find a better workaround so this can be pickled directly. Maybe __set_state__.
        with open(filename + '_parameters.pkl', 'wb') as f:
            parameters = self.get_params(),
            pickle.dump(parameters, f)
        with open(filename + '.pkl', 'wb') as f:
            objects = {'population': self.population_,
                       'logbook': self.logbook_,
                       'history': self.history_,
                       'best_individuals': self.best_individuals_}
            pickle.dump(objects, f)

    def load(self, filename):
        """Load a model.
        """
        with open(filename + '_parameters.pkl', 'rb') as f:
            parameters = pickle.load(f)
        self.set_params(**parameters[0])
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.experiment_ = self.experiment_class(ngen=self.ngen,
                                                 pop_size=self.pop_size,
                                                 tournament_size=self.tournament_size,
                                                 min_depth_init=self.min_depth_init,
                                                 max_dept_init=self.max_dept_init,
                                                 max_height=self.max_height,
                                                 max_size=self.max_size,
                                                 crossover_probability=self.crossover_probability,
                                                 mutation_probability=self.mutation_probability,
                                                 internal_node_selection_bias=self.internal_node_selection_bias,
                                                 min_gen_grow=self.min_gen_grow,
                                                 max_gen_grow=self.max_gen_grow,
                                                 subset_proportion=self.subset_proportion,
                                                 subset_change_frequency=self.subset_change_frequency,
                                                 num_randoms=self.num_randoms)
        self.pset_ = self.experiment_.get_pset(self.num_features, self.variable_type_indices, self.variable_names,
                                               self.variable_dict)
        # Create a temporary 'Fake' toolbox such that creator gets initialized with what it needs to create the
        # population. Hacky but fine.
        fake_features = np.zeros((1, self.num_features))
        fake_response = np.zeros((1, 1))
        self.experiment_.get_toolbox(fake_features, fake_response, self.pset_, self.variable_type_indices,
                                     self.variable_names)
        with open(filename + '.pkl', 'rb') as f:
            objects = pickle.load(f)
        self.population_ = objects['population']
        self.logbook_ = objects['logbook']
        self.history_ = objects['history']
        self.best_individuals_ = objects['best_individuals']

    def print_best_individuals(self):
        """Print the error and string representation of best_individuals_
        :return:
        """
        if self.best_individuals_ is None:
            raise RuntimeError('Cannot print best individuals. Model has not yet been fit!')
        for ind in self.best_individuals_:
            print(str(ind.error) + ' : ' + str(ind))
