import numpy as np


def trunc(num: float, trunc_at: float = 0, trunc_to: float = 0) -> float:
    if num < trunc_at:
        return trunc_to
    return num


def stretch_copy(arr: np.ndarray, scale: int = 5) -> np.ndarray:
    return arr.repeat(scale, axis=0)


def stretch_shift_add(arr2d: np.ndarray, scale: int = 5) -> np.ndarray:
    rows, cols = arr2d.shape
    res_cols = cols * scale

    arr2d_stretched = np.zeros((rows, res_cols))
    for i in range(cols):
        arr2d_stretched[:, i * scale] = arr2d[:, i]

    arr2d_shifted = np.zeros((rows, res_cols))
    digit_pos, digit_multi = -3, 10000
    shift_bits = (np.abs(arr2d[:, digit_pos]) * digit_multi).astype(int) % scale
    for i, shift in enumerate(shift_bits):
        arr2d_shifted[i] = np.roll(arr2d_stretched[i], shift)

    arr_added = arr2d_shifted.sum(axis=0)
    return arr_added


if __name__ == "__main__":
    arr1d = np.array([1.1, 2.2, 3.3])
    arr2d = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]])
    print(list(stretch_copy(arr1d, scale=5)))
    print(list(stretch_shift_add(arr2d, scale=5)))

    # python -m models.vectors.forms
