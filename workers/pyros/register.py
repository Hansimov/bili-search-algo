import numpy as np
import Pyro5.api

from workers.pyros.serialize import ndarray_to_dict, dict_to_ndarray


def register_pyro_apis():
    Pyro5.api.register_class_to_dict(np.ndarray, ndarray_to_dict)
    Pyro5.api.register_dict_to_class("numpy.ndarray", dict_to_ndarray)
