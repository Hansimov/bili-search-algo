import Pyro5.api
import numpy as np
import pyarrow as pa
from pyarrow.lib import StringScalar


NUMPY_GENERIC_NAME = "numpy.generic"
NUMPY_GENERIC_TYPE = np.generic
NUMPY_NDARRAY_NAME = "numpy.ndarray"
NUMPY_NDARRAY_TYPE = np.ndarray
PYARROW_STRING_NAME = "pyarrow.lib.StringScalar"
PYARROW_STRING_TYPE = StringScalar
NUMPY_DTYPE_NAME = "numpy.dtype"
NUMPY_DTYPE_TYPE = np.dtype


def serialize_numpy_generic(value: np.generic) -> dict:
    """serialize numpy generic value"""
    return {
        "__class__": "numpy.generic",
        "dtype": str(value.dtype),
        "value": value.item(),
    }


def construct_numpy_generic(classname: str, d: dict):
    """construct numpy generic value"""
    if "value" in d and "dtype" in d:
        return np.dtype(d["dtype"]).type(d["value"])
    else:
        return d


def serialize_numpy_dtype(dt: np.dtype) -> dict:
    """Serialize numpy.dtype to a dict"""
    return {
        "__class__": "numpy.dtype",
        "dtype": dt.name,
    }


def construct_numpy_dtype(classname: str, d: dict):
    """Construct numpy.dtype from dict"""
    return np.dtype(d["dtype"])


def serialize_numpy_ndarray(arr: np.ndarray) -> dict:
    """convert numpy.ndarray to dcit"""
    return {
        "__class__": "numpy.ndarray",  # must include this to notify Pyro to call dict_to_ndarray
        "__ndarray__": arr.tolist(),
        "dtype": str(arr.dtype),
        "shape": arr.shape,
    }


def construct_numpy_ndarray(classname: str, d: dict):
    """construct numpy.ndarray from dict"""
    if "__ndarray__" in d:
        return np.array(d["__ndarray__"], dtype=d["dtype"]).reshape(d["shape"])
    else:
        return d


def serialize_pyarrow_string(ss: StringScalar) -> dict:
    """serialize pyarrow.lib.StringScalar"""
    return {
        "__class__": "pyarrow.lib.StringScalar",
        "__str__": ss.as_py(),
    }


def construct_pyarrow_string(classname: str, d: dict):
    """construct pyarrow.lib.StringScalar"""
    if "__str__" in d:
        return pa.scalar(d["__str__"])
    else:
        return d


def register_pyro_apis():
    Pyro5.api.register_class_to_dict(NUMPY_GENERIC_TYPE, serialize_numpy_generic)
    Pyro5.api.register_dict_to_class(NUMPY_GENERIC_NAME, construct_numpy_generic)
    Pyro5.api.register_class_to_dict(NUMPY_NDARRAY_TYPE, serialize_numpy_ndarray)
    Pyro5.api.register_dict_to_class(NUMPY_NDARRAY_NAME, construct_numpy_ndarray)
    Pyro5.api.register_class_to_dict(NUMPY_DTYPE_TYPE, serialize_numpy_dtype)
    Pyro5.api.register_dict_to_class(NUMPY_DTYPE_NAME, construct_numpy_dtype)
    Pyro5.api.register_class_to_dict(PYARROW_STRING_TYPE, serialize_pyarrow_string)
    Pyro5.api.register_dict_to_class(PYARROW_STRING_NAME, construct_pyarrow_string)
