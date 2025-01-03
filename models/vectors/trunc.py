def trunc(num: float, trunc_at: float = 0, trunc_to: float = 0) -> float:
    if num < trunc_at:
        return trunc_to
    return num
