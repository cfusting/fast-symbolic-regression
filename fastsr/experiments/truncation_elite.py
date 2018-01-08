import math
import operator
import random
from functools import partial

import cachetools
import numpy
from deap import creator, base, tools, gp

from fastgp.algorithms import truncation_with_elite, fast_evaluate
from fastgp.logging import archive, reports
from fastgp.parametrized import simple_parametrized_terminals as sp
from fastgp.utilities import operators, symbreg, subset_selection, metrics
from fastsr.utilities import utils

NAME = 'TruncationElite'


def get_ephemeral():
    return random.gauss(0.0, 10.0)


class TruncationElite:

    def __init__(self,
                 ngen=50,
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
                 error_function=metrics.mean_squared_error,
                 num_randoms=1):

        super(TruncationElite, self).__init__()
        self.ngen = ngen
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.min_depth_init = min_depth_init
        self.max_depth_init = max_dept_init
        self.max_height = max_height
        self.max_size = max_size
        self.xover_prob = crossover_probability
        self.mut_prob = mutation_probability
        self.internal_node_selection_bias = internal_node_selection_bias
        self.min_gen_grow = min_gen_grow
        self.max_gen_grow = max_gen_grow
        self.subset_proportion = subset_proportion
        self.subset_change_frequency = subset_change_frequency
        self.error_function = error_function
        self.num_randoms = num_randoms
        self.log_mutate = False
        self.multi_archive = None
        self.pop = None
        self.mstats = None

    def get_toolbox(self, predictors, response, pset, variable_type_indices, variable_names):
        subset_size = int(math.floor(predictors.shape[0] * self.subset_proportion))
        creator.create("Error", base.Fitness, weights=(-1.0,))
        creator.create("Individual", sp.SimpleParametrizedPrimitiveTree, fitness=creator.Error, age=int)
        toolbox = base.Toolbox()
        toolbox.register("expr", sp.generate_parametrized_expression,
                         partial(gp.genHalfAndHalf, pset=pset, min_=self.min_depth_init, max_=self.max_depth_init),
                         variable_type_indices, variable_names)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("grow", sp.generate_parametrized_expression,
                         partial(gp.genGrow, pset=pset, min_=self.min_gen_grow, max_=self.max_gen_grow),
                         variable_type_indices, variable_names)
        toolbox.register("mutate", operators.mutation_biased, expr=toolbox.grow,
                         node_selector=operators.uniform_depth_node_selector)
        toolbox.decorate("mutate", operators.static_limit(key=operator.attrgetter("height"), max_value=self.max_height))
        toolbox.decorate("mutate", operators.static_limit(key=len, max_value=self.max_size))
        self.history = tools.History()
        toolbox.decorate("mutate", self.history.decorator)
        toolbox.register("error_func", self.error_function)
        expression_dict = cachetools.LRUCache(maxsize=1000)
        subset_selection_archive = subset_selection.RandomSubsetSelectionArchive(frequency=self.subset_change_frequency,
                                                                                 predictors=predictors,
                                                                                 response=response,
                                                                                 subset_size=subset_size,
                                                                                 expression_dict=expression_dict)
        evaluate_function = partial(subset_selection.fast_numpy_evaluate_subset,
                                    get_node_semantics=sp.get_node_semantics,
                                    context=pset.context,
                                    subset_selection_archive=subset_selection_archive,
                                    error_function=toolbox.error_func,
                                    expression_dict=expression_dict)
        toolbox.register("evaluate_error", evaluate_function)
        self.multi_archive = utils.get_archive(100)
        if self.log_mutate:
            mutation_stats_archive = archive.MutationStatsArchive(evaluate_function)
            toolbox.decorate("mutate", operators.stats_collector(archive=mutation_stats_archive))
            self.multi_archive.archives.append(mutation_stats_archive)
        self.multi_archive.archives.append(subset_selection_archive)
        self.mstats = reports.configure_parametrized_inf_protected_stats()
        self.pop = toolbox.population(n=self.pop_size)
        toolbox.register("run", truncation_with_elite.optimize, population=self.pop, toolbox=toolbox,
                         ngen=self.ngen, stats=self.mstats, archive=self.multi_archive, verbose=False,
                         history=self.history)
        toolbox.register("save", reports.save_log_to_csv)
        toolbox.decorate("save", reports.save_archive(self.multi_archive))
        return toolbox

    def get_best_individuals(self, population):
        return sorted(population, key=lambda x: x.error)

    def get_prediction_toolbox(self, features, pset):
        toolbox = base.Toolbox()
        toolbox.register("best_individuals", self.get_best_individuals)
        toolbox.register("predict",
                         fast_evaluate.fast_numpy_evaluate,
                         context=pset.context,
                         predictors=features,
                         get_node_semantics=sp.get_node_semantics)
        return toolbox

    def get_scoring_toolbox(self, features, response, pset):
        toolbox = base.Toolbox()
        toolbox.register("validate_func", partial(self.error_function, response=response))
        toolbox.register("score",
                         fast_evaluate.fast_numpy_evaluate,
                         get_node_semantics=sp.get_node_semantics,
                         context=pset.context,
                         predictors=features,
                         error_function=toolbox.validate_func)
        return toolbox

    def get_pset(self, num_predictors, variable_type_indices, names, variable_dict):
        pset = sp.SimpleParametrizedPrimitiveSet("MAIN", num_predictors, variable_type_indices, names)
        pset.addPrimitive(numpy.add, 2)
        pset.addPrimitive(numpy.subtract, 2)
        pset.addPrimitive(numpy.multiply, 2)
        pset.addPrimitive(symbreg.numpy_protected_log_abs, 1)
        pset.addPrimitive(numpy.exp, 1)
        pset.addPrimitive(numpy.cbrt, 1)
        pset.addPrimitive(symbreg.cube, 1)
        pset.addPrimitive(symbreg.numpy_protected_sqrt, 1)
        pset.addPrimitive(numpy.square, 1)
        pset.renameArguments(**variable_dict)
        return pset


