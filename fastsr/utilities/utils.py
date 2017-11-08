import glob
import importlib

from sklearn import preprocessing

from fastgp.logging import archive


def get_pareto_files(results_path, experiment, data_set_name):
    return glob.glob(results_path + "/pareto_*_po_{}_*.log".format(get_identifier(data_set_name, experiment)))


def get_experiment(experiment):
    mod = importlib.import_module("experiments." + experiment)
    return getattr(mod, mod.NAME)()


def get_identifier(data_set, experiment):
    return data_set + '_' + experiment


def get_archive(saving_frequency):
    pareto_archive = archive.ParetoFrontSavingArchive(frequency=saving_frequency,
                                                      criteria_chooser=archive.
                                                      pick_fitness_size_complexity_from_fitness_age_size_complexity)
    multi_archive = archive.MultiArchive([pareto_archive])
    return multi_archive


def transform_features(predictors, response, predictor_transformer=None, response_transformer=None):
    if predictor_transformer is None and response_transformer is None:
        predictor_transformer = preprocessing.StandardScaler()
        response_transformer = preprocessing.StandardScaler()
    predictors_transformed = predictor_transformer.fit_transform(predictors, response)
    response_transformed = response_transformer.fit_transform(response)
    return predictors_transformed, response_transformed, predictor_transformer, response_transformer


