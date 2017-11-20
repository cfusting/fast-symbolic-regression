import math
import operator
import random
from functools import partial

import cachetools
import numpy
from deap import creator, base, tools, gp

from fastgp.algorithms import afpo, fast_evaluate
from fastgp.logging import archive, reports
from fastgp.parametrized import simple_parametrized_terminals as sp
from fastgp.utilities import operators, symbreg, subset_selection, metrics
from fastsr.experiments import abstract_experiment
from fastsr.utilities import utils

NAME = 'Control'


def get_ephemeral():
    return random.gauss(0.0, 10.0)


class Control(abstract_experiment.Experiment):

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

        super(Control, self).__init__()
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
        self.algorithm_names = ["afsc_po"]
        self.num_randoms = num_randoms
        self.log_mutate = False
        self.multi_archive = None
        self.pop = None
        self.mstats = None

    def get_toolbox(self, predictors, response, pset, variable_type_indices, variable_names):
        subset_size = int(math.floor(predictors.shape[0] * self.subset_proportion))
        creator.create("ErrorAgeSizeComplexity", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
        creator.create("Individual", sp.SimpleParametrizedPrimitiveTree, fitness=creator.ErrorAgeSizeComplexity,
                       age=int)
        toolbox = base.Toolbox()
        toolbox.register("expr", sp.generate_parametrized_expression,
                         partial(gp.genHalfAndHalf, pset=pset, min_=self.min_depth_init, max_=self.max_depth_init),
                         variable_type_indices, variable_names)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("select", tools.selRandom)
        toolbox.register("koza_node_selector", operators.internally_biased_node_selector,
                         bias=self.internal_node_selection_bias)
        self.history = tools.History()
        toolbox.register("mate", operators.one_point_xover_biased, node_selector=toolbox.koza_node_selector)
        toolbox.decorate("mate", operators.static_limit(key=operator.attrgetter("height"), max_value=self.max_height))
        toolbox.decorate("mate", operators.static_limit(key=len, max_value=self.max_size))
        toolbox.decorate("mate", self.history.decorator)
        toolbox.register("grow", sp.generate_parametrized_expression,
                         partial(gp.genGrow, pset=pset, min_=self.min_gen_grow, max_=self.max_gen_grow),
                         variable_type_indices, variable_names)
        toolbox.register("mutate", operators.mutation_biased, expr=toolbox.grow,
                         node_selector=toolbox.koza_node_selector)
        toolbox.decorate("mutate", operators.static_limit(key=operator.attrgetter("height"), max_value=self.max_height))
        toolbox.decorate("mutate", operators.static_limit(key=len, max_value=self.max_size))
        toolbox.decorate("mutate", self.history.decorator)

        def generate_randoms(individuals):
            return individuals
        toolbox.register("generate_randoms", generate_randoms,
                         individuals=[toolbox.individual() for i in range(self.num_randoms)])
        toolbox.decorate("generate_randoms", self.history.decorator)
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
        toolbox.register("assign_fitness", afpo.assign_age_fitness_size_complexity)
        self.multi_archive = utils.get_archive(100)
        if self.log_mutate:
            mutation_stats_archive = archive.MutationStatsArchive(evaluate_function)
            toolbox.decorate("mutate", operators.stats_collector(archive=mutation_stats_archive))
            self.multi_archive.archives.append(mutation_stats_archive)
        self.multi_archive.archives.append(subset_selection_archive)
        self.mstats = reports.configure_parametrized_inf_protected_stats()
        self.pop = toolbox.population(n=self.pop_size)
        toolbox.register("run", afpo.pareto_optimization, population=self.pop, toolbox=toolbox,
                         xover_prob=self.xover_prob, mut_prob=self.mut_prob, ngen=self.ngen,
                         tournament_size=self.tournament_size,  num_randoms=self.num_randoms, stats=self.mstats,
                         archive=self.multi_archive, calc_pareto_front=False, verbose=False, reevaluate_population=True,
                         history=self.history)
        toolbox.register("save", reports.save_log_to_csv)
        toolbox.decorate("save", reports.save_archive(self.multi_archive))
        return toolbox

    def get_best_individuals(self, population):
        indices = list(afpo.find_pareto_front(population))
        return [population[i] for i in indices]

    def get_prediction_toolbox(self, features, pset):
        toolbox = base.Toolbox()
        toolbox.register("best_individuals", afpo.find_pareto_front)
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
        # pset.addEphemeralConstant("gaussian", get_ephemeral)
        pset.renameArguments(**variable_dict)
        return pset


