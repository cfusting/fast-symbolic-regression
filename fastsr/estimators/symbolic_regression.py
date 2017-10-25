import random
import pickle

from sklearn.base import BaseEstimator

import fastsr.data.learning_data as ld
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


class SymbolicRegression(BaseEstimator):

    def __init__(self,
                 experiment_class=None,
                 variable_type_indices=None,
                 variable_names=None,
                 variable_dict=None,
                 num_features=None,
                 ngen=2,
                 pop_size=10,
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
        self.variable_names = ['x' + str(x) for x in range(self.num_features)]
        self.variable_dict = ld.get_variable_dict(self.variable_names, ld.LearningData.DEFAULT_PREFIX)

    def fit(self, X, y):
        check_data(X)
        check_data(y)
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
        return self

    def predict(self, X):
        check_data(X)
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
        check_data(X)
        check_data(y)
        if self.num_features != X.shape[1]:
            raise ValueError('Cannot score model with a different number of features than with what the model was '
                             'fit.')
        scoring_toolbox = self.experiment_.get_scoring_toolbox(X, y, self.pset_)
        scores = np.zeros(len(self.best_individuals_))
        for i, ind in enumerate(self.best_individuals_):
            scores[i] = scoring_toolbox.score(ind)[0]
        return scores.min()

    def save(self, filename):
        with open(filename + '_parameters', 'wb') as f:
            parameters = self.get_params(),
            pickle.dump(parameters, f)
        with open(filename, 'wb') as f:
            objects = {'population': self.population_,
                       'logbook': self.logbook_,
                       'history': self.history_,
                       'best_individuals': self.best_individuals_}
            pickle.dump(objects, f)

    def load(self, filename):
        with open(filename + '_parameters', 'rb') as f:
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
        with open(filename, 'rb') as f:
            objects = pickle.load(f)
        self.population_ = objects['population']
        self.logbook_ = objects['logbook']
        self.history_ = objects['history']
        self.best_individuals_ = objects['best_individuals']
