import ntpath
import os
import re
from functools import partial

import h5py

import fastsr.data.design_matrix as dm


class LearningData:
    DEFAULT_PREFIX = 'ARG'
    CSV = '.csv'
    HDF = '.hdf'

    def __init__(self):
        self.num_variables = None
        self.num_observations = None
        self.predictors = None
        self.response = None
        self.variable_names = None
        self.unique_variable_prefixes = None
        self.variable_type_indices = None
        self.variable_dict = None
        self.name = None
        self.design_matrix = None
        self.attributes = {}
        self.meta_layers = {}

    def from_file(self, file_name, header=True):
        file_type = os.path.splitext(file_name)[1]
        if file_type == self.HDF:
            self.from_hdf(file_name)
        elif file_type == self.CSV and header:
            self.from_headed_csv(file_name)
        elif file_type == self.CSV:
            self.from_csv(file_name)
        else:
            raise ValueError("Bad file: " + file_name + ". File extension must be one of csv, hdf.")

    def init_common(self, file_name):
        self.name = os.path.splitext(ntpath.basename(file_name))[0]
        self.predictors = self.design_matrix.predictors
        self.response = self.design_matrix.response
        self.num_observations, self.num_variables = self.predictors.shape
        self.variable_names = self.design_matrix.variable_names
        self.unique_variable_prefixes = get_unique_variable_prefixes(self.variable_names)
        variable_groups = get_variable_groups(self.variable_names, self.unique_variable_prefixes)
        self.variable_type_indices = get_variable_type_indices(variable_groups)
        self.variable_dict = get_variable_dict(self.variable_names, self.DEFAULT_PREFIX)

    def from_csv(self, csv_file):
        self.design_matrix = dm.DesignMatrix()
        self.design_matrix.from_csv(csv_file)
        self.init_common(csv_file)

    def from_headed_csv(self, csv_file):
        self.design_matrix = dm.DesignMatrix()
        self.design_matrix.from_headed_csv(csv_file)
        self.init_common(csv_file)

    def from_hdf(self, hdf_file):
        self.design_matrix = dm.DesignMatrix()
        self.design_matrix.from_hdf(hdf_file)
        self.init_common(hdf_file)
        self.get_meta_layers(hdf_file)
        self.get_layer_attributes(hdf_file, 'design_matrix')

    def from_data(self, matrix, variable_names, name):
        self.design_matrix = dm.DesignMatrix()
        self.design_matrix.from_data(matrix, variable_names)
        self.init_common(name)

    def to_hdf(self, file_name):
        self.design_matrix.to_hdf(file_name)
        self.save_meta_layers(file_name)
        self.save_layer_attributes(file_name, 'design_matrix')

    def to_headed_csv(self, file_name):
        self.design_matrix.to_headed_csv(file_name)

    def get_meta_layers(self, file_name):
        with h5py.File(file_name, 'r') as f:
            layers = filter(lambda x: x != 'design_matrix', f.keys())
            for layer in layers:
                self.meta_layers[layer] = f[layer][:]

    def save_meta_layers(self, file_name):
        with h5py.File(file_name, 'r+') as f:
            for k, v in self.meta_layers.items():
                f.create_dataset(k, data=v)

    def get_layer_attributes(self, file_name, layer):
        with h5py.File(file_name, 'r') as f:
            dset = f[layer]
            for k, v in dset.items():
                self.attributes[k] = v

    def save_layer_attributes(self, file_name, layer):
        with h5py.File(file_name, 'r+') as f:
            dset = f[layer]
            for k, v in self.attributes.items():
                dset.attrs[k] = v


def get_variable_dict(names, default_prefix):
    args = [default_prefix + str(x) for x in range(0, len(names))]
    return dict(zip(args, names))


def get_variable_type_indices(variable_groups):
    indices = []
    previous = 0
    for i in variable_groups:
        current = previous + len(i)
        if len(i) != 0:
            indices.append(current - 1)
            previous = current
    return indices


def get_unique_variable_prefixes(variable_names):
    """
    Assumes the form prefixnumber.
    :param variable_names:
    :return:
    """
    expr = re.compile('([a-zA-Z]+)')

    def get_prefix(name, expression):
        result = re.match(expression, name)
        if result:
            return result.group(1)
        return ''
    prefixes = list(set(list(map(partial(get_prefix, expression=expr), variable_names))))
    return prefixes


def get_variable_groups(variable_names, unique_prefixes):
    variable_groups = []
    for prefix in unique_prefixes:
        variable_groups.append(list(filter(lambda x: prefix in x, variable_names)))
    return variable_groups
