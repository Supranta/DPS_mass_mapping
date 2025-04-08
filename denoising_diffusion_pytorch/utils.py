import numpy as np

def string_to_numpy_array(comma_separated_string):
    return np.array([float(x) for x in comma_separated_string.split(',')])
