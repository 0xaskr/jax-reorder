# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Matrix operations built on top of torch to support the math in the library.

The functions here intentionally mirror the names in the accompanying docs.
They operate on 1D/2D tensors and keep broadcasting explicit for clarity.
"""

from typing import List, Tuple, Union

import sympy as sp

Matrix = sp.Matrix


def is_column_vector(x: Matrix) -> bool:
    m, n = x.shape
    return n == 1


def is_row_vector(x: Matrix) -> bool:
    m, n = x.shape
    return m == 1


def vector_size(x: Matrix) -> int:
    return max(x.shape)


def is_square(X: Matrix) -> bool:
    """Check if X is a square matrix."""
    m, n = X.shape
    return m == n


def dot(x: Matrix, y: Matrix):
    """Dot product of vectors x and y."""
    if is_column_vector(x) and is_column_vector(y):
        return (x.T * y)[0, 0]
    elif is_row_vector(x) and is_row_vector(y):
        return (x * y.T)[0, 0]
    raise RuntimeError('dot: received illegal input')


def zeros(m: int, n: int = 1) -> Matrix:
    """
    Returns an mxn matrix with all elements equal to 0.
    """
    return sp.zeros(m, n)


def ones(m: int, n: int = 1) -> Matrix:
    """
    Returns an mxn matrix with all elements equal to 1.
    """
    return sp.ones(m, n)


def identity(n: int) -> Matrix:
    """
    Returns the nxn identity matrix.
    """
    return sp.eye(n)


def product(X: Matrix, Y: Matrix) -> Matrix:
    """Matrix multiplication X * Y."""
    return X * Y


def hadamard(X: Matrix, Y: Matrix) -> Matrix:
    """Element-wise product X ⊙ Y."""
    assert X.shape == Y.shape
    m, n = X.shape
    return Matrix([[X[i, j] * Y[i, j] for j in range(n)] for i in range(m)])
    # return matrix_multiply_elementwise(X, Y)  ==> this may cause errors:
    #     def mul_elementwise(A, B):
    # >       assert A.domain == B.domain
    # E       AssertionError


def diag(X: Matrix) -> Matrix:
    """Extract diagonal of X as a vector."""
    assert is_square(X)
    m, n = X.shape
    return Matrix([[X[i, i] for i in range(m)]]).T


def Diag(x: Matrix) -> Matrix:
    """Create diagonal matrix with x as diagonal."""
    assert is_column_vector(x) or is_row_vector(x)
    return sp.diag(*x)


def elements_sum(X: Matrix):
    """
    Returns the sum of the elements of X.
    """
    m, n = X.shape
    return sum(X[i, j] for i in range(m) for j in range(n))


def column_repeat(x: Matrix, n: int) -> Matrix:
    """Repeat column vector x horizontally n times."""
    assert is_column_vector(x)
    rows, cols = x.shape
    rows = [[x[i, 0]] * n for i in range(rows)]
    return Matrix(rows)


def row_repeat(x: Matrix, m: int) -> Matrix:
    """Repeat row vector x vertically m times."""
    assert is_row_vector(x)
    rows, cols = x.shape
    columns = [[x[0, j]] * m for j in range(cols)]
    return Matrix(columns).T


def columns_sum(X: Matrix) -> Matrix:
    """Sum over columns (returns row vector)."""
    m, n = X.shape
    columns = [sum(X.col(j)) for j in range(n)]
    return Matrix(columns).T


def rows_sum(X: Matrix) -> Matrix:
    """Sum over rows (returns column vector)."""
    m, n = X.shape
    rows = [sum(X.row(i)) for i in range(m)]
    return Matrix(rows)


def columns_max(X: Matrix) -> Matrix:
    """
    Returns a column vector with the maximum values of each row in X.
    """
    m, n = X.shape
    return Matrix([[sp.Max(*X.col(j)) for j in range(n)]])


def rows_max(X: Matrix) -> Matrix:
    """
    Returns a row vector with the maximum values of each column in X.
    """
    m, n = X.shape
    return Matrix([[sp.Max(*X.row(i)) for i in range(m)]]).T


def columns_mean(X: Matrix) -> Matrix:
    """
    Returns a column vector with the mean values of each row in X.
    """
    m, n = X.shape
    return columns_sum(X) / m


def rows_mean(X: Matrix) -> Matrix:
    """
    Returns a row vector with the mean values of each column in X.
    """
    m, n = X.shape
    return rows_sum(X) / n


def apply(f, X: Matrix) -> Matrix:
    """Element-wise application of function f to X."""
    return X.applyfunc(f)


def exp(X: Matrix) -> Matrix:
    """Element-wise exponential exp(X)."""
    return X.applyfunc(sp.exp)


def log(X: Matrix) -> Matrix:
    """Element-wise natural logarithm log(X)."""
    return X.applyfunc(sp.log)


def reciprocal(X: Matrix) -> Matrix:
    """Element-wise reciprocal 1/X."""
    return X.applyfunc(lambda x: 1 / x)


def square(X: Matrix) -> Matrix:
    """Element-wise square X²."""
    return X.applyfunc(lambda x: x * x)


def sqrt(X: Matrix) -> Matrix:
    """Element-wise square root √X."""
    return X.applyfunc(sp.sqrt)


def inv_sqrt(X: Matrix) -> Matrix:
    """Element-wise inverse square root X^(-1/2)."""
    return reciprocal(sqrt(X))


def log_sigmoid(X: Matrix) -> Matrix:
    """Element-wise log(sigmoid(X))."""
    # N.B. We don't provide a numerically stable version in nerva-sympy
    from nerva_sympy.activation_functions import Sigmoid
    return log(Sigmoid(X))


def join_columns(columns: List[Matrix]) -> Matrix:
    assert all(is_column_vector(column) for column in columns)
    return Matrix([x.T for x in columns]).T


def join_rows(rows: List[Matrix]) -> Matrix:
    assert all(is_row_vector(row) for row in rows)
    return Matrix(rows)


def gradient(f, X) -> Matrix:
    """
    Returns the derivative of a matrix function
    :param f: a real-valued function
    :param X: a matrix
    :return: the derivative of f
    """
    if not isinstance(X, sp.MatrixBase):
        X = sp.Matrix([[X]])
    m, n = X.shape
    return Matrix([[sp.diff(f, X[i, j]) for j in range(n)] for i in range(m)])


def substitute(expr, substitutions: Union[Tuple[Matrix, Matrix], List[Tuple[Matrix, Matrix]]]):
    if isinstance(substitutions, tuple):
        substitutions = [substitutions]
    for (X, Y) in substitutions:
        assert X.shape == Y.shape
        m, n = X.shape
        sigma = ((X[i, j], Y[i, j]) for i in range(m) for j in range(n))
        expr = expr.subs(sigma)
    return expr


def jacobian(x: Matrix, y) -> Matrix:
    assert is_column_vector(x) or is_row_vector(x)
    return x.jacobian(y)
