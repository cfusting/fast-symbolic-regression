from fastgp.experiments import symbreg


class Feature:

    def __init__(self, constructor, prefixes):
        self.name = None
        self.representation = None
        self.constructor = constructor
        self.score = None
        self.prefixes = prefixes

    def from_infix_string(self, feature_string):
        self.name = ''.join(feature_string.rstrip().split())
        self.representation = self.constructor(infix_to_prefix(feature_string, self.prefixes))


def infix_to_prefix(infix_string, prefix):
    symbol_map = symbreg.get_numpy_prefix_symbol_map()
    prefix_equation = symbreg.get_prefix_from_infix(infix_string, prefix)
    for k, v in symbol_map:
        prefix_equation = prefix_equation.replace(k + "(", v + "(")
    return prefix_equation
