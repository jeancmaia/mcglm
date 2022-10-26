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
    inverse_power,
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
    "inverse_power": inverse_power(),
    "reciprocal": Power(),
}


class MCGLMMean(MCGLMCAttributes):
    """
    MCGLMMean is the class for the first moment adjustment within MCGLM inference. It handles lifecycle completely, ranging from mu and derivatives to quasi-likelihood calculations.
    This class has two interfaces: 'calculate_mean_features' which calculates mu attributes, and 'update_beta' that applies quasi-likelihood estimation and retrieves a new beta.
    This class implements the Estimating Equation Quasi-score (Wedderburn, 1974) and the second-order optimization algorithm (Jennrich, 1969) and(Widyaningsih et al., 2017).

    References
    ----------
    Wedderburn, R. W. M. (1974). Quasi-likelihood functions, generalized linear models, and the Gauss—Newton method. Biometrika, 61(3):439–447.

    Jennrich, R. I. (1969). A Newton-Raphson algorithm for maximum likelihood factor analysis. Psychometrika, 34.

    Widyaningsih, P., Saputro, D. e Putri, A. (2017). Fisher scoring method for parameter estimation of geographically weighted ordinal logistic regression (gwolr) model. Journal of Physics: Conference Series, 855:012060.
    """

    def __init__(self):
        super(MCGLMCAttributes, self).__init__()

    def __get_link_function(self, link: str):
        """Check whether the link function is available

        Parameters
        ----------
            link : str
                A link function
        Returns
        -------
            statsmodels.genmod.families.links : a corresponding object for the link function.
        """
        assert link in AVAILABLE_LINK_FUNCTIONS, (
            f"The link function " + str(link) + " isn't available"
        )
        return AVAILABLE_LINK_FUNCTIONS.get(link.lower())

    def __linear_predictor(self, X, beta):
        """Method linear predictor applies linear operation between covariates and the regression parameters.

        Parameters
        ----------
            X : array_like
                Design matrix with covariates
            beta : array-like
                Regression parameters
        Returns
        -------
            array_like : The calculated output vector.
        """
        return np.dot(X, beta)

    def _link_function_attributes(
        self, link: str, beta: np.array, X: np.array, offset: int = 0
    ):
        """A protected method for calling the __link_function_attributes

        Parameters
        ----------
            link : str
                Link function.
            beta : array_like
                Regression Parameters.
            X : array_like
                Matrix with covariates.
            offset : (int, optional)
                Offset add value. Defaults to 0.
        Returns
        -------
            dict : Dict value with the mean and its derivatives.
        """
        return self.__link_function_attributes(link, beta, X, offset)

    def __link_function_attributes(
        self, link: str, beta: np.array, X: np.array, offset: int = 0
    ):
        """The method __link_function_attributes calculates the vector of expected values and its derivatives. It returns data as dictionary

        Parameters
        ----------
            link : str
                Link function.
            beta : array-like
                Regression Parameters.
            X : array-like
                Matrix with covariates.
            offset : int, optional
                Offset add value. Defaults to 0.
        Returns
        -------
            dict : Dict value with the mean and its derivatives.
        """
        link_func = self.__get_link_function(link.lower())
        eta = self.__linear_predictor(X, beta)
        if offset is None:
            offset = 0
        eta = eta + offset

        mu = link_func.inverse(eta)
        deriv = X * link_func.inverse_deriv(eta)[:, None]
        return dict(mu=mu, deriv=deriv)

    def calculate_mean_features(self, link, beta, X, offset):
        """Base method to calculate every attribute related to the mean.

        Parameters
        ----------
            link : str
                Link function.
            beta : array-like
                Regression Parameters.
            X : array-like
                Matrix with covariates.
            offset : int, optional)
                Offset add value. Defaults to 0.
        Returns
        -------
            tuple : Mean attributes, the raw mean and its derivatives.
        """
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
        """The method update_beta takes the current beta, leverages the quasi-likelihood estimator to calculate the next regression parameters.

        Parameters
        ----------
            beta : array-like
                Regression Parameters.
            W : array-like
                Weight matrix
            power : float
                Power parameter
            rho : float
                Correlation parameters
            tau : float
                Dispersion parameters.
        Returns
        -------
            tuple : A tuple with the new regression parameters, the quasi-score parameter, sensitivity and the variability matrix.
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
        """A private method that implements the second-order optimization algorithm Fisher-scoring.

        Parameters
        ----------
            sensitivity : array-like
                Sensitivity matrix
            score : array-like
                Quasi-score output.
            beta : array-like
                Regression parameters.

        Returns
        -------
            array-like : New regression vector.
        """
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
        """Quasi-score functions are estimating equations for the Maximum Likelihood Estimator (Wedderburn, 1974). This method harnesses the Numpy backbone to produce a classic method of statistical models.

        Parameters
        ----------
            mu_derivative : array-like
                First-order derivatives of means.
            c_inverse : array-like
                Inverse of C matrix
            y : array-like
                A vector with outcome variable.
            mu : array-like
                A vetor with mean parameters.
            W : array-like
                A matrix of weights.

        Returns
        -------
            tuple : A tuple with score, sensitivity and variability.
        """

        residue = y - mu

        mu_derivative_transpose = mu_derivative.transpose()
        mu_derivative_and_c = np.dot(mu_derivative_transpose, c_inverse)

        score = np.dot(np.dot(mu_derivative_and_c, W), residue)
        sensitivity = np.dot(np.dot(-mu_derivative_and_c, W), mu_derivative)
        variability = np.dot(np.dot(mu_derivative_and_c, W**2), mu_derivative)

        return (score, sensitivity, variability)
