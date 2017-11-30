import fastsr.containers.learning_data as ld

import numpy as np
import numpy.testing as npt


class TestLearningData:

    def test_get_unique_variable_prefixes(self):
        variable_names = ['hour20tdlta0', 'temptdlta10', 'wind', 'hum9']
        assert ld.get_unique_variable_prefixes(variable_names) == list(set(['hour20', 'temp', 'wind', 'hum9']))

    def test_get_variable_type_indices(self):
        names = ['cat']
        assert ld.get_variable_type_indices(names) == []
        names = ['cattdlta0']
        assert ld.get_variable_type_indices(names) == []
        names = ['cattdlta0', 'cattdlta1']
        assert ld.get_variable_type_indices(names) == [(0, 2)]
        names = ['cattdlta0', 'cattdlta1', 'dog']
        assert ld.get_variable_type_indices(names) == [(0, 2)]
        names = ['cattdlta0', 'cattdlta1', 'dog', 'ferrettdlt0']
        assert ld.get_variable_type_indices(names) == [(0, 2)]
        names = ['cattdlta0', 'cattdlta1', 'dog', 'ferrettdlt0', 'penguintdlta0', 'penguintdlta1']
        assert ld.get_variable_type_indices(names) == [(0, 2), (4, 6)]

    def test_generate_lagged_column_names(self):
        column_names = []
        new_names = ld.generate_lagged_column_names(column_names, 1, 1)
        assert new_names == []
        column_names = ['x']
        new_names = ld.generate_lagged_column_names(column_names, 1, 1)
        assert new_names == ['xtdlta0', 'xtdlta1']
        column_names = ['x', 'y']
        new_names = ld.generate_lagged_column_names(column_names, 1, 1)
        assert new_names == ['xtdlta0', 'xtdlta1', 'ytdlta0', 'ytdlta1']

    def test_generate_lagged_column_names_with_indices(self):
        column_names = ['x', 'y']
        new_names = ld.generate_lagged_column_names(column_names, 1, 1, column_indices=[1])
        assert new_names == ['x', 'ytdlta0', 'ytdlta1']

    def test_lag_vec(self):
        mat = np.array([1, 2, 3])
        conv_matrix = ld.lag_vector(mat, 1, 1)
        result = [[2, 1], [3, 2]]
        npt.assert_array_equal(conv_matrix, result)

    def test_generate_lagged_matrix(self):
        mat = np.array([[1, 7], [2, 8], [3, 9]])
        conv_matrix = ld.generate_lagged_matrix(mat, 1, 1)
        result = [[2, 1, 8, 7], [3, 2, 9, 8]]
        npt.assert_array_equal(conv_matrix, result)

    def test_generate_lagged_matrix_with_indices(self):
        mat = np.array([[1, 7], [2, 8], [3, 9]])
        conv_matrix = ld.generate_lagged_matrix(mat, 1, 1, columnn_indices=[0])
        result = [[2, 1, 7], [3, 2, 8]]
        npt.assert_array_equal(conv_matrix, result)
