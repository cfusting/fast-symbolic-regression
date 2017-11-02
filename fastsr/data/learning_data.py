import ntpath
import os
import re
from functools import partial
import utilities

import h5py

import fastsr.data.design_matrix as dm


class LearningData:
    DEFAULT_PREFIX = 'ARG'
    CSV = '.csv'
    HDF = '.hdf'
    TIME_DELTA = 'tdlta'
    EXPRESSION = re.compile('([a-zA-Z0-9]+)' + TIME_DELTA)

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
        # self.attributes = {}
        self.meta_layers = {}

    def from_file(self, file_name, header=True):
        file_type = os.path.splitext(file_name)[1]
        self.name = os.path.splitext(ntpath.basename(file_name))[0]
        if file_type == self.HDF:
            self.from_hdf(file_name)
        elif file_type == self.CSV and header:
            self.from_headed_csv(file_name)
        elif file_type == self.CSV:
            self.from_csv(file_name)
        else:
            raise ValueError("Bad file: " + file_name + ". File extension must be one of csv, hdf.")

    def init_common(self):
        self.predictors = self.design_matrix.predictors
        self.response = self.design_matrix.response
        self.num_observations, self.num_variables = self.predictors.shape
        self.variable_names = self.design_matrix.variable_names
        self.unique_variable_prefixes = get_unique_variable_prefixes(self.variable_names)
        self.variable_type_indices = get_variable_type_indices(self.variable_names)
        self.variable_dict = get_variable_dict(self.variable_names, self.DEFAULT_PREFIX)

    def from_csv(self, csv_file):
        self.design_matrix = dm.DesignMatrix()
        self.design_matrix.from_csv(csv_file)
        self.init_common()

    def from_headed_csv(self, csv_file):
        self.design_matrix = dm.DesignMatrix()
        self.design_matrix.from_headed_csv(csv_file)
        self.init_common()

    def from_hdf(self, hdf_file):
        self.design_matrix = dm.DesignMatrix()
        self.design_matrix.from_hdf(hdf_file)
        self.init_common()
        self.get_meta_layers(hdf_file)
        # self.get_layer_attributes(hdf_file, 'design_matrix')

    def from_data(self, matrix, variable_names, name):
        self.name = name
        self.design_matrix = dm.DesignMatrix()
        self.design_matrix.from_data(matrix, variable_names)
        self.init_common()

    def to_hdf(self, file_name):
        self.design_matrix.to_hdf(file_name)
        self.save_meta_layers(file_name)
        # self.save_layer_attributes(file_name, 'design_matrix')

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
    #
    # def get_layer_attributes(self, file_name, layer):
    #     with h5py.File(file_name, 'r') as f:
    #         dset = f[layer]
    #         for k, v in dset.items():
    #             self.attributes[k] = v
    #
    # def save_layer_attributes(self, file_name, layer):
    #     with h5py.File(file_name, 'r+') as f:
    #         dset = f[layer]
    #         for k, v in self.attributes.items():
    #             dset.attrs[k] = v

    def lag_predictors(self, lag, every=1, column_names=None):
        def get_matrix_and_names(indices):
            cols = utilities.generate_lagged_column_names(self.variable_names, lag, every,
                                                          indices)
            dt = utilities.generate_lagged_matrix(self.design_matrix.dat, lag, every, indices)
            return cols, dt
        if column_names is None:
            idx = [x for x in range(len(self.variable_names) - 1)]
            column_names, dat = get_matrix_and_names(idx)
        else:
            idx = []
            for i, j in enumerate(self.variable_names):
                if j in column_names:
                    idx.append(i)
            column_names, dat = get_matrix_and_names(idx)
        self.design_matrix.from_data(dat, column_names)
        self.init_common()


def get_prefix(name):
        if LearningData.TIME_DELTA not in name:
            return name
        result = re.match(LearningData.EXPRESSION, name)
        if result:
            return result.group(1)
        return ''


def get_variable_type_indices(variable_names):
    """
    Return the indexes of variables that have a range over time.
    :param variable_names:
    :return:
    """
    variable_type_indices = []
    begin_index = 0
    current_prefix = get_prefix(variable_names[begin_index])
    for i, name in enumerate(variable_names[1:] + ['']):
        prefix = get_prefix(name)
        delta_index = i - begin_index
        if current_prefix != prefix and delta_index > 0:
            variable_type_indices.append((begin_index, i + 1))
            begin_index = i + 1
        elif current_prefix != prefix:
            begin_index = i + 1
        current_prefix = prefix
    return variable_type_indices


def get_variable_dict(names, default_prefix):
    args = [default_prefix + str(x) for x in range(0, len(names))]
    return dict(zip(args, names))


def get_unique_variable_prefixes(variable_names):
    """
    Assumes variables of the form: NAMEtdltaTIMESTEPS. Where NAME is alphanumeric and TIMESTEPS is numeric.
    :param variable_names:
    :return:
    """
    prefixes = list(set(list(map(partial(get_prefix), variable_names))))
    return prefixes
