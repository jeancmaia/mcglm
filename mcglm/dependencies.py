"""
An extension of MCGLM library to provide three options for matrix linear predictor: mc_id, mc_ma, and mc_mixed.
"""
import numpy as np
import itertools

from patsy import dmatrix
from mcglm.utils import diagonal


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
    design_matrix = dmatrix(formula, data=data, return_type="dataframe")
    val_columns = design_matrix.columns.tolist()

    positions = list()
    for column in val_columns:
        if ":" in column:
            positions.append(1)
        else:
            positions.append(0)
    positions = np.array(positions)

    design_matrix = design_matrix.values

    all_indexes = [(i, i) for i in set(positions)]

    if len(all_indexes) > 1:
        all_indexes = all_indexes + list(itertools.combinations(list(range(2)), 2))

    matrices = list()
    for tc in all_indexes:
        first_m = design_matrix[:, np.where(positions == tc[0])[0]]
        second_m = design_matrix[:, np.where(positions == tc[1])[0]]

        if tc[0] == tc[1]:
            matrices.append(np.matmul(first_m, second_m.T))
        else:
            matrices.append(
                np.matmul(first_m, second_m.T) + np.matmul(second_m, first_m.T)
            )

    return matrices
