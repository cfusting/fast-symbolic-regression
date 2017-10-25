from abc import ABCMeta, abstractmethod


class Experiment:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def get_toolbox(self, predictors, response, pset, variable_type_indices, variable_names):
        pass

    @abstractmethod
    def get_best_individuals(self, population):
        pass

    @abstractmethod
    def get_prediction_toolbox(self, features, pset):
        pass

    @abstractmethod
    def get_scoring_toolbox(self, features, response, pset):
        pass

    @abstractmethod
    def get_pset(self, num_predictors, variable_type_indices, variable_names, variable_dict):
        pass



