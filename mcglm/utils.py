import numpy as np
from functools import reduce


def diagonal(n: int, values: np.array):
    raw_matrix = np.zeros((n, n), float)
    np.fill_diagonal(raw_matrix, list(values))
    return raw_matrix


def mc_sandwich(central_matrix, left_matrix, right_matrix):
    return np.dot(np.dot(left_matrix, central_matrix), right_matrix)


def mc_sandwich_power(central_matrix, left_matrix, right_matrix):
    matrix_operation = mc_sandwich(central_matrix, left_matrix, right_matrix)
    return matrix_operation + matrix_operation.transpose()


def mc_matrix_linear_predictor(tau: list, z: list):
    map_operation = map(lambda x, y: x * y, tau, z)
    liner_predictor_calculus = reduce(lambda x, y: x + y, map_operation)
    return liner_predictor_calculus


def mc_sandwich_csr(central_matrix, left_matrix, right_matrix):
    return left_matrix.dot(central_matrix).dot(right_matrix)


def mc_sandwich_power_csr(central_matrix, left_matrix, right_matrix):
    matrix_operation = mc_sandwich_csr(central_matrix, left_matrix, right_matrix)
    return matrix_operation + matrix_operation.transpose()
