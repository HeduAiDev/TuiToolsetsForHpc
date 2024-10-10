from ._C import *

try:
    import numpy as np
except ImportError:
    raise RuntimeError('numpy is required for tui_tool_sets')

def _ensure_ndarray(matrix):
    import sys
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    # import torch is time consuming
    elif 'torch' in sys.modules and isinstance(matrix, sys.modules['torch'].Tensor):
        matrix = matrix.cpu().numpy()
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a list, numpy.ndarray or torch.Tensor')
    return matrix

def print_matrix(matrix):
    matrix = _ensure_ndarray(matrix)
    if matrix.ndim != 2:
        raise ValueError('matrix must be a 2D array')
    if matrix.dtype == np.float64:
        print_matrix_double(matrix)
    elif matrix.dtype == np.float32:
        print_matrix_float(matrix)
    elif matrix.dtype == np.float16:
        try:
            print_matrix_half(matrix)
        except Exception:
            print_matrix_float(matrix.astype(np.float32))
    elif matrix.dtype in (np.int16, np.int32, np.int64):
        print_matrix_float(matrix.astype(np.float32))
    else:
        raise ValueError(f'unsupported matrix dtype: {matrix.dtype}')
            
def diff(a, b, accuracy: float = 1e-3):
    a = _ensure_ndarray(a)
    b = _ensure_ndarray(b)

    if a.ndim != 2:
        raise ValueError('matrix "a" must be a 2D array')
    if b.ndim != 2:
        raise ValueError('matrix "b" must be a 2D array')
    if a.shape != b.shape:
        raise ValueError('matrix "a" and "b" must have the same shape')
    if a.dtype != b.dtype:
        b = b.astype(a.dtype)

    if a.dtype == np.float64:
        diff_double(a, b, accuracy)
    elif a.dtype == np.float32:
        diff_float(a, b, accuracy)
    elif a.dtype == np.float16:
        try:
            diff_half(a, b, accuracy)
        except Exception:
            diff_float(a.astype(np.float32), b.astype(np.float32), accuracy)
    elif a.dtype in (np.int16, np.int32, np.int64):
        diff_float(a.astype(np.float32), b.astype(np.float32), accuracy)
    else:
        raise ValueError(f'unsupported matrix dtype: {a.dtype}')
    
__all__ = [
    'print_matrix',
    'diff'
]