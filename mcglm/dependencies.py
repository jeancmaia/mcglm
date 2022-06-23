"""
An extension of MCGLM library to provide three options for matrix linear predictor: mc_id, mc_ma, and mc_mixed.
"""
import numpy as np
import itertools

from patsy import dmatrix
from mcglm.utils import diagonal
from itertools import combinations


def mc_id(data=None):
    """
    mc_id method retrieves a numpy diagonal matrix with data length of the original matrix
    """
    size = data.shape[0]
    return diagonal(size, np.ones(size))


def mc_ma(id=None, time=None, data=None, order=1):
    """
    mc_ma method retrieves the components for matrix linear predictor associated with moving average models.
    """

    def diagonal_position(indexes=None, k=1):
        new_indexes = tuple((indexes[0][:-k] + k, indexes[1][:-k]))
        return new_indexes

    try:
        data = data.sort_values(by=[id, time], ascending=True)
    except Exception as e:
        print("Parameter data must be a pandas DataFrame")
        raise e

    min_time = sorted(np.unique(data[time]))[:order]
    id = data[id].values
    time = data[time].values

    sample_size = len(id)
    ma = np.zeros((sample_size, sample_size))

    ma_vector = np.ones(sample_size - order)
    index_time = sorted(
        list(itertools.chain(*[np.where(time == value)[0] for value in min_time]))
    )

    for index in index_time[order:]:
        ma_vector[index - order] = 0

    indexes_pos = np.diag_indices(sample_size)
    indexes = diagonal_position(indexes_pos, order)

    ma[indexes] = ma_vector

    return ma + ma.T


def mc_mixed(data=None, formula=None):
    """
    mc_mixed retrieves the components for matrix linear predictor associated with mixed models.
    """
    design_matrix = dmatrix(formula, data=data, return_type="dataframe").values

    positions = list(range(design_matrix.shape[1]))
    size = design_matrix.shape[0]

    matrices = list()

    all_indexes = [(i, i) for i in positions] + list(combinations(list(range(2)), 2))

    for tc in all_indexes:

        matrix1 = np.repeat(design_matrix[:, tc[0]], size, axis=0).reshape(size, size).T
        matrix2 = np.repeat(design_matrix[:, tc[1]], size, axis=0).reshape(size, size).T

        if tc[0] == tc[1]:
            matrices.append(np.multiply(matrix1, matrix2.T))
        else:
            matrices.append(
                np.multiply(matrix1, matrix2.T) + np.multiply(matrix2, matrix1.T)
            )

    return matrices
