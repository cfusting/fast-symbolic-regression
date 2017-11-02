import fastsr.data.learning_data as ld


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
