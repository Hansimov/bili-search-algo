import numpy as np


def dot_sim(v1: np.ndarray, v2: np.ndarray, ndigits: int = None) -> float:
    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if ndigits:
        sim = round(sim, ndigits)
    return sim
