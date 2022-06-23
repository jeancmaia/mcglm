import numpy as np
import itertools

from .mcglmcattr import MCGLMCAttributes
from statsmodels.genmod.families.links import (
    Logit,
    Power,
    Log,
    probit,
    cauchy,
    CLogLog,
    LogLog,
    NegativeBinomial,
)
from scipy.linalg import block_diag
from numpy.linalg import solve


AVAILABLE_LINK_FUNCTIONS = {
    "logit": Logit(),
    "identity": Power(power=1.0),
    "power": Power(),
    "log": Log(),
    "probit": probit(),
    "cauchy": cauchy(),
    "cloglog": CLogLog(),
    "loglog": LogLog(),
    "negativebinomial": NegativeBinomial(),
}


class MCGLMMean(MCGLMCAttributes):
    """
    MCGLMMean is the class for the first moment adjustment within MCGLM inference. It handles lifecycle completely, ranging from mu and derivatives to quasi-likelihood calculations.
    This class has two interfaces: 'calculate_mean_features' which calculates mu attributes, and 'update_beta' that applies quasi-likelihood estimation and retrieves a new beta.
    """

    def __init__(self):
        super(MCGLMCAttributes, self).__init__()

    def __get_link_function(self, link: str):
        assert link in AVAILABLE_LINK_FUNCTIONS, (
            f"The link function " + str(link) + " isn't available"
        )
        return AVAILABLE_LINK_FUNCTIONS.get(link.lower())

    def __linear_predictor(self, X, beta):
        return np.dot(X, beta)

    def _link_function_attributes(
        self, link: str, beta: np.array, X: np.array, offset: int = 0
    ):
        return self.__link_function_attributes(link, beta, X, offset)

    def __link_function_attributes(
        self, link: str, beta: np.array, X: np.array, offset: int = 0
    ):
        link_func = self.__get_link_function(link.lower())
        eta = self.__linear_predictor(X, beta)
        if offset is None:
            offset = 0
        eta = eta + offset

        mu = link_func.inverse(eta)
        deriv = X * link_func.inverse_deriv(eta)[:, None]
        return dict(
            mu=mu, deriv=deriv
        )  # TODO: pickup a pythonic implementation for it.

    def calculate_mean_features(self, link, beta, X, offset):
        mu_attributes_per_response = list(
            map(self.__link_function_attributes, link, beta, X, offset)
        )
        mu = np.array(
            list(
                itertools.chain(
                    *[mu_value.get("mu") for mu_value in mu_attributes_per_response]
                )
            )
        )
        d = block_diag(
            *[mu_value.get("deriv") for mu_value in mu_attributes_per_response]
        )
        return mu_attributes_per_response, mu, d

    def update_beta(self, beta, W, power, rho, tau):
        """
        update_beta takes current beta, calculates quasi-likelihood estimation and returns the next beta.
        """
        mu_attributes, mu, derivative_mu = self.calculate_mean_features(
            self._link, beta, self._X, self._offset
        )
        c_inverse = self.c_inverse(mu_attributes, power, rho, tau)

        score, sensitivity, variability = self.__quasi_score(
            derivative_mu, c_inverse, self._y_values, mu, W
        )
        new_beta = self.__update_fisher_score(sensitivity, score, beta)

        return new_beta, score, sensitivity, variability

    def __update_fisher_score(self, sensitivity, score, beta):

        linear_system = solve(sensitivity, score)
        index = 0
        adjusted_betas = []
        for beta_value in beta:
            adjusted_betas.append(linear_system[index : index + len(beta_value)])
            index += len(beta_value)
        adjusted_betas = np.array(adjusted_betas)
        new_beta = beta - adjusted_betas

        return new_beta

    def __quasi_score(self, mu_derivative, c_inverse, y, mu, W):
        """
        Quasi-score method optimization has been odopted due to its adaptability and flexibility in comparision to the classical maximum likelihood.
        """

        residue = y - mu

        mu_derivative_transpose = mu_derivative.transpose()
        mu_derivative_and_c = np.dot(mu_derivative_transpose, c_inverse)

        score = np.dot(np.dot(mu_derivative_and_c, W), residue)
        sensitivity = np.dot(np.dot(-mu_derivative_and_c, W), mu_derivative)
        variability = np.dot(np.dot(mu_derivative_and_c, W**2), mu_derivative)

        return (score, sensitivity, variability)
