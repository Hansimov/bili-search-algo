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


def stretch_shift_add(arr2d: np.ndarray, scale: int = 5, offset: int = 0) -> np.ndarray:
    rows, cols = arr2d.shape
    res_cols = cols * scale

    # stretch with inserting zeros
    arr2d_stretched = np.zeros((rows, res_cols))
    arr2d_stretched[:, ::scale] = arr2d

    # shift with step cols
    arr2d_shifted = np.zeros((rows, res_cols))
    # shift_bits = random_shift_bits(arr2d, scale, digit_pos=100, digit_multi=1000)
    shift_bits = step_shift_bits(rows=rows, scale=scale)
    if offset is not None and offset > 0:
        shift_bits += offset
    for i in range(rows):
        arr2d_shifted[i] = np.roll(arr2d_stretched[i], shift_bits[i])

    # add the shifted rows
    arr_added = arr2d_shifted.sum(axis=0)
    return arr_added


def padding_zeros(arr: np.ndarray, padding_len: int) -> np.ndarray:
    arr_cols = arr.shape[1] if arr.ndim > 1 else len(arr)
    if arr.ndim == 1:
        if padding_len == arr_cols:
            arr_padded = arr
        else:
            arr_padded = np.pad(arr, (0, padding_len - arr_cols), mode="constant")
    else:
        if padding_len == arr_cols:
            arr_padded = arr
        else:
            arr_padded = np.pad(
                arr, ((0, 0), (0, padding_len - arr_cols)), mode="constant"
            )
    return arr_padded


def calc_padded_downsampled_cols(cols: int, nume_deno: tuple[int, int]) -> int:
    nume, deno = nume_deno
    if nume == deno:
        return cols
    else:
        return (cols + deno - 1) // deno * nume


def downsample(
    arr: np.ndarray,
    ratio: float = None,
    to_num: int = None,
    nume_deno: tuple[int, int] = None,
    method: Literal["successive", "step", "window"] = "window",
) -> np.ndarray:
    if ratio is None and to_num is None and nume_deno is None:
        return arr

    arr_cols = arr.shape[1] if arr.ndim > 1 else len(arr)
    arr_rows = arr.shape[0] if arr.ndim > 1 else 1

    if nume_deno is None and to_num is None and ratio is not None:
        to_num = int(arr_cols * ratio)

    if method == "step" and arr_cols >= to_num * 2:
        step = arr_cols // to_num
        if arr.ndim == 1:
            return arr[::step]
        else:
            return arr[:, ::step]
    elif method == "window":
        nume, deno = nume_deno
        if nume == deno:
            return arr
        # padding zeros to make arr cols be divided by deno
        padding_len = (arr_cols + deno - 1) // deno * deno
        if arr.ndim == 1:
            arr_padded = padding_zeros(arr, padding_len)
            return arr_padded.reshape(-1, deno)[:, :nume].flatten()
        else:
            arr_padded = padding_zeros(arr, padding_len)
            return arr_padded.reshape(arr_rows, -1, deno)[:, :, :nume].reshape(
                arr_rows, -1
            )
    else:
        if arr.ndim == 1:
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
    from tclogger import logger

    logger.note("arr1d:")
    arr1d = np.array([1.1, 2.2, 3.3])
    print(list(arr1d))

    logger.note("stretch_copy:")
    print(list(stretch_copy(arr1d, scale=5)))

    logger.note("arr2d:")
    arr2d = np.array(
        [
            [1.1, 2.2, 3.3, 4.4],
            [4.2, 5.3, 6.4, 7.5],
            [7.3, 8.4, 9.5, 10.6],
        ]
    )
    print(list(arr2d))

    logger.note("step_shift_bits:")
    print(step_shift_bits(rows=3, scale=5))
    logger.note("step_shift_bits with offset:")
    print(step_shift_bits(rows=3, scale=5) + 1)

    logger.note("stretch_shift_add:")
    print(list(stretch_shift_add(arr2d, scale=5)))

    logger.note("arr1d2:")
    arr1d2 = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    print(list(arr1d2))

    logger.note("downsample arr1d2:")
    print(list(downsample(arr1d2, nume_deno=(2, 3), method="window")))

    logger.note("downsample arr2d:")
    print(downsample(arr2d, nume_deno=(1, 1), method="window"))

    logger.note("calc_padded_downsampled_cols:")
    print(calc_padded_downsampled_cols(320, (3, 3)))

    # python -m models.vectors.forms
