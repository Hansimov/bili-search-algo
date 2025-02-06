import numpy as np

from typing import Literal


def trunc(num: float, trunc_at: float = 0, trunc_to: float = 0) -> float:
    if num < trunc_at:
        return trunc_to
    return num


def stretch_copy(arr: np.ndarray, scale: int = 5) -> np.ndarray:
    return arr.repeat(scale, axis=0)


def random_shift_bits(
    arr2d: np.ndarray, scale: int, digit_pos: int = 10, digit_multi: int = 100
) -> np.ndarray:
    return (np.abs(arr2d[:, digit_pos]) * digit_multi).astype(int) % scale


def step_shift_bits(rows: int, scale: int) -> np.ndarray:
    # the shift bits would have rows elements
    # return np.array([int(scale * i / rows) for i in range(rows)])
    return np.floor((np.arange(rows) * scale) / rows).astype(int)


def stretch_shift_add(arr2d: np.ndarray, scale: int = 5) -> np.ndarray:
    rows, cols = arr2d.shape
    res_cols = cols * scale

    # stretch with inserting zeros
    arr2d_stretched = np.zeros((rows, res_cols))
    arr2d_stretched[:, ::scale] = arr2d

    # shift with step cols
    arr2d_shifted = np.zeros((rows, res_cols))
    # shift_bits = random_shift_bits(arr2d, scale, digit_pos=100, digit_multi=1000)
    shift_bits = step_shift_bits(rows, scale)
    for i in range(rows):
        arr2d_shifted[i] = np.roll(arr2d_stretched[i], shift_bits[i])

    # add the shifted rows
    arr_added = arr2d_shifted.sum(axis=0)
    return arr_added


def downsample(
    arr: np.ndarray,
    ratio: float = None,
    to_num: int = None,
    method: Literal["successive", "step"] = "step",
) -> np.ndarray:
    if ratio is None and to_num is None:
        return arr

    arr_dims = len(arr.shape)
    arr_cols = arr.shape[1] if arr_dims > 1 else len(arr)

    if to_num is None and ratio is not None:
        to_num = int(arr_cols * ratio)

    if method == "step" and arr_cols >= to_num * 2:
        step = arr_cols // to_num
        if arr_dims == 1:
            return arr[::step]
        else:
            return arr[:, ::step]
    else:
        if arr_dims == 1:
            return arr[:to_num]
        else:
            return arr[:, :to_num]


def sample_to_dim(arr: np.ndarray, dim: int, n_window: int) -> np.ndarray:
    # select [:dim] for every window length arr parts
    # example: arr = [1, 2, 3, 4, 5, 6, 7, 8, 9], dim = 2, window = 3
    # result = [1, 2, 4, 5, 7, 8]
    window = len(arr) // n_window
    arr_reshaped = arr[: n_window * window].reshape(-1, window)
    return arr_reshaped[:, :dim].flatten()


if __name__ == "__main__":
    arr1d = np.array([1.1, 2.2, 3.3])
    print(list(arr1d))
    print(list(stretch_copy(arr1d, scale=5)))
    arr2d = np.array([[1.1, 2.2, 3.3], [4.2, 5.3, 6.4], [7.3, 8.4, 9.5]])
    print(list(arr2d))
    print(list(stretch_shift_add(arr2d, scale=5)))
    arr1d2 = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    print(list(downsample(arr1d2, to_num=2)))

    # python -m models.vectors.forms
