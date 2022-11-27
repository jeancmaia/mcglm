import numpy as np

from numpy.linalg import solve
from .mcglmcattr import MCGLMCAttributes


class MCGLMVariance(MCGLMCAttributes):
    """
    The MCGLMVariance class handles the second optimization of the MCGLM second-moment assumptions, therefore, the step for variance. It implements every step of Variance within the scope of the MCGLM algorithm, using many attributes to be specified as attributes. A general class must inherit this MCGLMVariance and leverages its methods properly. MCGLM is in charge of setting the fundamental python attributes and modules orchestration for a complete mcglm adjustment.

    The variance step on the optimization sketch boils down to Pearson estimating equations and the chaser algorithm for optimization. The latter uses tuning to set the step size of each iteration.

    Heavy operations regarding C components, pivotal to the chaser optimization step, are implemented on the MCGLMAttricutes class, inherited here. The method _c_complete, the one that crafts all of the three attributes thoroughly, is comprehensive for the variance calculation in this class.
    """

    def __init__(self):
        super(MCGLMCAttributes, self).__init__()

    def update_covariates(self, mu_attributes, rho, power, tau, W, dispersion, mu):
        """The method update_covariates implements a cycle of iteration for the second-moment estimation, the variance.

        Parameters
        ----------
            mu_attributes : dict
                A dict with mean and derivatives.
            rho : array_type
                Parameters of correlation.
            power : float
                A parameter for Power Tweedie distribution.
            tau : float
                Dispersion parameters.
            W : array_type
                A weight matrix.
            dispersion : array-type
                A vector with dispersion parameters.
            mu : array_type
                A vector with mean parameters.
        Returns
        -------
            tuple: A tuple with new vector of dispersion vector, atributes of matrix C, sensitivity.
        """
        c_inverse, c_derivative, c_values = self.c_complete(
            mu_attributes, power, rho, tau
        )
        (
            pearson_score,
            sensitivity,
            c_intermediate_components,
        ) = self.__pearson_estimating_equation(mu, W, c_inverse, c_derivative)

        step = self.__chaser_step(pearson_score, sensitivity)
        new_covariates = dispersion - step

        return (
            new_covariates,
            c_inverse,
            c_values,
            c_intermediate_components,
            sensitivity,
        )

    def __chaser_step(self, score, sensitivity):
        """The protected method chaser step calculates the step for a optimization step.

        Parameters
        ----------
            score : array_type
                A vector with quasi-score values.
            sensitivity : array_type
                The sensitivity matrix.
        Returns
        -------
            array_type: The absolute change to operate on vector.
        """
        return self._tuning * solve(sensitivity, score)

    def __pearson_estimating_equation(self, mu, W, c_inverse, c_derivative):
        """The estimating equation for the dispersion parameters.

        Parameters
        ----------
            mu : array_type
                A vetor with mean parameters
            W : array_type
                A weight matrix
            c_inverse : array_type
                The inverse of C Matrix
            c_derivative : array_type
                The derivatives of C Matrix
        Returns
        -------
            tuple : a tuple with score, sensitivity matrix and the matrix C normalized by Pearson.
        """
        residue = self._y_values - mu

        c_pearson = [np.dot(c_inverse, d_c) for d_c in c_derivative]

        pearson_score = [
            self.__core_pearson(matrix_component, c_inverse, residue, W)
            for matrix_component in c_pearson
        ]
        sensitivity = self.generate_sensitivity(c_pearson, W)
        return (pearson_score, sensitivity, c_pearson)

    def __core_pearson(self, c_component, c_inverse, residue, W):
        """The protected method core_pearson handles the inner-components of Pearson estimation equations operations.

        Parameters
        ----------
            c_component : array_type
                A matrix with components of C.
            c_inverse : array_type
                The inverse of matrix C.
            residue : array_type
                The difference between mean and an outcome variable.
            W : array_type
                A weight matrix.
        Returns
        -------
            array_type : A matrix with the core pearson.
        """
        weighted_c_components = np.dot(c_component, W)
        sum_diagonal = np.sum(np.diag(weighted_c_components))
        residue_inv_C = np.dot(c_inverse, residue)

        core_pearson = np.subtract(
            np.dot(np.dot(residue.transpose(), weighted_c_components), residue_inv_C),
            sum_diagonal,
        )
        return core_pearson

    @staticmethod
    def generate_sensitivity(c_intermediate_components, W):
        """The method to create the sensitivity matrix.

        Parameters
        ----------
            c_intermediate_components : array_type
                Intermediate components of C matrix.
            W : array_type
                A weight matrix.
        Returns
        -------
            array_type : A sensitivity matrix
        """
        sensitivity = np.array([])

        for position_row in range(len(c_intermediate_components)):
            transpose_matrix_position = (
                c_intermediate_components[position_row].copy().transpose()
            )
            transpose_matrix_position = np.dot(W, transpose_matrix_position)
            for position_col in range(len(c_intermediate_components)):
                sensitivity = np.append(
                    sensitivity,
                    -np.sum(
                        np.multiply(
                            transpose_matrix_position,
                            c_intermediate_components[position_col],
                        )
                    ),
                )
        sensitivity = sensitivity.reshape(
            len(c_intermediate_components), len(c_intermediate_components)
        )
        return sensitivity
