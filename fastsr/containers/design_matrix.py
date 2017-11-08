import numpy

import h5py


class DesignMatrix:

    def __init__(self):
        self.dat = None
        self.predictors = None
        self.response = None
        self.variable_names = None

    def from_hdf(self, hdf_file):
        with h5py.File(hdf_file, 'r') as f:
            dset = f['design_matrix']
            self.dat = numpy.nan_to_num(dset[:])
            self.set_predictors_and_response()
            if dset.attrs['variable_names']:
                self.variable_names = dset.attrs['variable_names'].split(",")
            else:
                self.generate_simple_variable_names()

    def from_csv(self, csv_file):
        self.dat = numpy.nan_to_num(numpy.genfromtxt(csv_file, delimiter=',', skip_header=True))
        self.set_predictors_and_response()
        self.generate_simple_variable_names()

    def from_headed_csv(self, csv_file):
        self.dat = numpy.nan_to_num(numpy.genfromtxt(csv_file, dtype=numpy.float, delimiter=',', names=True,
                                    deletechars="""~!@#$%^&-=~\|]}[{';: /?.>,<"""))
        self.variable_names = list(self.dat.dtype.names[:-1])
        self.dat = self.dat.view((numpy.float, len(self.dat.dtype.names)))
        self.set_predictors_and_response()

    def from_data(self, matrix, variable_names):
        self.dat = numpy.nan_to_num(matrix)
        self.set_predictors_and_response()
        if variable_names:
            self.variable_names = variable_names
        else:
            self.generate_simple_variable_names()

    def to_hdf(self, file_name):
        with h5py.File(file_name, 'w') as f:
            dset = f.create_dataset('design_matrix', data=self.dat)
            dset.attrs['variable_names'] = ",".join(self.variable_names)

    def to_headed_csv(self, file_name):
        numpy.savetxt(file_name, X=self.dat, delimiter=',', header=','.join(self.variable_names))

    def set_predictors_and_response(self):
        self.predictors = self.dat[:, :-1]
        self.response = self.dat[:, -1]

    def generate_simple_variable_names(self):
        self.variable_names = ['X' + str(x) for x in range(0, self.predictors.shape[1])]
