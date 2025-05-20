import numpy as np

def matrix_poly_sum(A: np.ndarray, k:int) -> np.ndarray:
    result = np.eye(A.shape[0], dtype=A.dtype)
    Ak = np.eye(A.shape[0], dtype=A.dtype)
    for _ in range(k):
        Ak = Ak @ A
        result += Ak
    return result