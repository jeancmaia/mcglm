import numpy as np
import itertools


from scipy.sparse import csr_matrix, tril, kron
from scipy.linalg import block_diag
from itertools import combinations
from functools import lru_cache
from numpy.linalg import inv, cholesky
from .utils import (
    mc_sandwich,
    diagonal,
    mc_sandwich_power,
    mc_matrix_linear_predictor,
    mc_sandwich_power_csr,
)


class MCGLMCAttributes:
    """
    The class "MCGLMCAttributes" has the task of calculating every C operations, used on throughout adjustments of mean and variance. This class has two interfaces, "c_inverse" and "c_complete"; one for each of two adjustment steps of MCGLM.

    The interface "c_inverse" crafts only inverse C, and the "c_complete" adds its derivatives and other features onto response. A Quasi-likelihood estimation needs only the inverse of "C" matrix. Therefore c_inverse saves computational resources by avoiding unnecessary operations on mean step adjustment.
    """

    def c_inverse(self, mu, power, rho, tau, full_response=False):
        """
        A protected method to calculate inverse of C matrix, made specifically for quasi likelihood step. This method interacts with sigma and omega methods, passing the list of each parameter.

        -----
        params:

        responds:

        """
        c_inverse = self.__generate_c_inverse(mu, power, rho, tau, full_response)
        return c_inverse

    def c_complete(self, mu, power, rho, tau):
        """
        Full response
        """
        (
            diagonal_matrix,
            omega,
            sigma_raw,
            sigma_chol,
            sigma_chol_inv,
            sigma_between,
            sigma_between_derivative,
            sigma_chol_block_matrix,
            sigma_chol_inv_block_matrix,
            c_inverse,
        ) = self.__generate_c_inverse(mu, power, rho, tau, full_response=True)

        sigma_derivatives = self.__generate_c_derivatives(omega, mu, power)

        core_matrix = np.kron(sigma_between, diagonal_matrix)
        sigma_chol_block_matrix_transpose = sigma_chol_block_matrix.transpose()
        c_derivatives = self.__generate_derivative_cov(
            sigma_chol,
            sigma_chol_inv,
            sigma_derivatives,
            core_matrix,
            sigma_chol_block_matrix,
            sigma_chol_block_matrix_transpose,
            sigma_between_derivative,
            diagonal_matrix,
            self._n_targets,
        )
        c_values = self.__generate_c_values(
            sigma_between,
            diagonal_matrix,
            sigma_chol_block_matrix_transpose,
            sigma_chol_block_matrix,
        )

        return c_inverse, c_derivatives, c_values

    def __generate_c_inverse(self, mu, power, rho, tau, full_response=False):
        """This method generates and retrieves the inverse of the C matrix, which is a pivotal component of the quasi-score calculation. Notwithstanding the mentioned, this method crafts many important artifacts throughout the time as omega, sigma, sigma Cholesky, the inverse of Cholesky sigma, sigma among responses, sigma block diagonal matrix alongside its inverse.
        Owing to the previously mentioned worthy artifacts crafted by the method, it can retrieve either only the c inverse matrix or all artifacts. The former is helpful for the method _c_inverse, whereas the former is for _c_complete.
        """
        diagonal_matrix = diagonal(self._n_obs, np.ones(self._n_obs))

        omega = self._generate_omega(tau)
        build_sigma = map(
            self._calculate_sigma,
            mu,
            power,
            omega,
            self._variance,
            self._ntrial,
        )
        sigma_raw, sigma_chol, sigma_chol_inv = self.__parser_sigma(build_sigma)
        sigma_between, sigma_between_derivative = self._sigma_between_values(
            rho=rho, n_resp=self._n_targets
        )
        raw_c_components = self.__generate_c_inverse_and_blocks(
            sigma_chol, sigma_chol_inv, sigma_between, diagonal_matrix
        )

        (
            sigma_chol_block_matrix,
            sigma_chol_inv_block_matrix,
            c_inverse,
        ) = self.__parser_c_inverse(raw_c_components)

        if not full_response:
            return c_inverse
        else:
            return (
                diagonal_matrix,
                omega,
                sigma_raw,
                sigma_chol,
                sigma_chol_inv,
                sigma_between,
                sigma_between_derivative,
                sigma_chol_block_matrix,
                sigma_chol_inv_block_matrix,
                c_inverse,
            )

    def __generate_c_derivatives(self, omega, mu, power):

        build_sigma = map(
            self._calculate_sigma_derivatives,
            mu,
            power,
            self._variance,
            self._z,
            self._power_fixed,
            self._ntrial,
            omega,
        )

        sigma_derivative = self.__parser_sigma_derivatives(build_sigma)
        return sigma_derivative

    def __generate_derivative_cov(
        self,
        sigma_chol,
        sigma_chol_inv,
        sigma_derivatives,
        core_matrix,
        sigma_chol_block_matrix,
        sigma_chol_block_matrix_transpose,
        sigma_between_derivative,
        diagonal_matrix,
        n_target,
    ):

        if n_target == 1:

            core_matrix_csr = sigma_between_derivative
            sigma_chol_block_matrix_csr = sigma_chol_block_matrix
            sigma_chol_block_matrix_transpose_csr = csr_matrix(
                sigma_chol_block_matrix_transpose
            )
            sigma_between_derivative_csr = sigma_between_derivative

        else:
            core_matrix_csr = csr_matrix(core_matrix)
            sigma_chol_block_matrix_csr = csr_matrix(sigma_chol_block_matrix)
            sigma_chol_block_matrix_transpose_csr = csr_matrix(
                sigma_chol_block_matrix_transpose
            )
            sigma_between_derivative_csr = []
            for sigma in sigma_between_derivative:
                sigma_between_derivative_csr.append(csr_matrix(sigma))

            diagonal_matrix_csr = csr_matrix(diagonal_matrix)

        list_d_chol_sigma = list(
            map(
                self.__derivative_cholesky,
                sigma_derivatives,
                sigma_chol_inv,
                sigma_chol,
            )
        )
        bdiag_d_chol_sigma = list(
            map(
                self.__mc_transform_list_bdiag,
                list_d_chol_sigma,
                list(range(n_target)),
                [n_target for _ in range(n_target + 1)],
            )
        )
        bdiag_d_chol_sigma_csr = []
        for sigmas in bdiag_d_chol_sigma:
            bdiag_d_chol_sigma_csr.append([csr_matrix(sigma) for sigma in sigmas])
        if n_target == 1:
            bdiag_d_chol_sigma_csr = bdiag_d_chol_sigma_csr[0]
            d_c = [
                mc_sandwich_power_csr(
                    core_matrix_csr,
                    d_chol_sigma,
                    sigma_chol_block_matrix_transpose_csr,
                ).toarray()
                for d_chol_sigma in bdiag_d_chol_sigma_csr
            ]
            D_C = d_c
        else:
            bdiag_d_chol_sigma = list(itertools.chain(*list(bdiag_d_chol_sigma_csr)))

            d_c = [
                mc_sandwich_power_csr(
                    core_matrix_csr,
                    d_chol_sigma,
                    sigma_chol_block_matrix_transpose_csr,
                ).toarray()
                for d_chol_sigma in bdiag_d_chol_sigma
            ]
            d_c_rho = self.__derivative_rho(
                sigma_between_derivative_csr,
                sigma_chol_block_matrix_csr,
                sigma_chol_block_matrix_transpose_csr,
                diagonal_matrix_csr,
            )
            D_C = d_c_rho + d_c
        return D_C

    def __generate_c_values(
        self,
        sigma_between,
        diagonal_matrix,
        sigma_chol_block_matrix_transpose,
        sigma_chol_block_matrix,
    ):
        kron_middle = np.kron(sigma_between, diagonal_matrix)

        c_computation = np.dot(
            np.dot(
                sigma_chol_block_matrix_transpose,
                kron_middle,
            ),
            sigma_chol_block_matrix,
        )
        return c_computation

    def __derivative_rho(
        self,
        d_sigmab,
        sigma_chol_block_matrix,
        sigma_chol_block_matrix_transpose,
        core_matrix,
    ):
        def multiplication_matrix(
            sigmab,
            sigma_chol_block_matrix,
            sigma_chol_block_matrix_transpose,
            core_matrix,
        ):
            kron_matrix = kron(sigmab, core_matrix)
            return (sigma_chol_block_matrix_transpose.dot(kron_matrix)).dot(
                sigma_chol_block_matrix
            )

        return [
            multiplication_matrix(
                sigmab,
                sigma_chol_block_matrix,
                sigma_chol_block_matrix_transpose,
                core_matrix,
            ).toarray()
            for sigmab in d_sigmab
        ]

    def _generate_omega(self, tau):
        """
        It builds omega matrix for covariance link setted as identity.
        """
        omega = []
        for target in range(self._n_targets):
            omega.append(mc_matrix_linear_predictor(tau=tau[target], z=self._z[target]))

        return omega

    def __generate_c_inverse_and_blocks(
        self,
        sigma_chol: list,
        sigma_chol_inv: list,
        sigma_between: np.array,
        diagonal_matrix,
    ):
        """ """
        sigma_chol_block_matrix = block_diag(*sigma_chol)
        sigma_chol_inv_block_matrix = block_diag(*sigma_chol_inv)

        try:
            ls_sigma_diagonal = np.linalg.inv(sigma_between)
        except Exception as e:
            ls_sigma_diagonal = np.array([[1]])

        C_inverse = np.dot(
            np.dot(
                sigma_chol_inv_block_matrix,
                np.kron(ls_sigma_diagonal, diagonal_matrix),
            ),
            sigma_chol_inv_block_matrix.transpose(),
        )

        return (
            sigma_chol_block_matrix,
            sigma_chol_inv_block_matrix,
            C_inverse,
        )

    def _calculate_sigma(
        self, mu, power, omega, variance, Ntrial, covariance="identity"
    ):
        """
        Base method for computing variance-covariance matrix, based on variance function and omega matrix. This method will implement for cases where covariance is equal to identity, and variance falls in the list:
        ['constant', 'tweedie', 'binomialP', 'binomialPQ', 'power', 'geom_tweedie', 'poisson_tweedie']
        """
        if isinstance(variance, list):
            variance = variance[0]
        if variance == "constant":
            sigma_raw = omega
            sigma_chol = cholesky(sigma_raw).T
            sigma_chol_inv = inv(sigma_chol)
        elif variance in ["tweedie", "binomialP", "binomialPQ", "power"]:
            if variance == "tweedie":
                variance = "power"

            variance_components = self.__generate_variance(
                variance_type=variance, mu=mu.get("mu"), power=power, Ntrial=Ntrial
            )
            sigma_raw = mc_sandwich(
                omega,
                variance_components.get("variance_sqrt_output"),
                variance_components.get("variance_sqrt_output"),
            )
            sigma_chol = cholesky(sigma_raw).T
            sigma_chol_inv = inv(sigma_chol)
        elif variance in ["poisson_tweedie", "geom_tweedie"]:
            diagonal_value = [mu.get("mu") ** 2, mu.get("mu")][
                variance == "poisson_tweedie"
            ]

            variance_components = self.__generate_variance(
                variance_type="power", mu=mu.get("mu"), power=power, Ntrial=Ntrial
            )
            sigma_raw = diagonal(len(mu.get("mu")), diagonal_value) + np.dot(
                np.dot(variance_components.get("variance_sqrt_output"), omega),
                variance_components.get("variance_sqrt_output"),
            )
            sigma_chol = cholesky(sigma_raw).T
            sigma_chol_inv = inv(sigma_chol)
        return dict(
            sigma_raw=sigma_raw, sigma_chol=sigma_chol, sigma_chol_inv=sigma_chol_inv
        )

    def _calculate_sigma_derivatives(
        self, mu, power, variance, z, power_fixed, Ntrial, omegas, covariance="identity"
    ):
        """
        Base method for computing variance-covariance matrix, based on variance function and omega matrix. This method will implement for cases where covariance is equal to identity, and variance falls in the list:
        ['constant', 'tweedie', 'binomialP', 'binomialPQ', 'power', 'geom_tweedie', 'poisson_tweedie']
        """
        if isinstance(variance, list):
            variance = variance[0]
        sigma_derivative = None

        if variance == "constant":
            sigma_derivative = z
        elif variance in ["tweedie", "binomialP", "binomialPQ", "power"]:
            if variance == "tweedie":
                variance = "power"
            variance_components = self.__generate_variance(
                variance_type=variance, mu=mu.get("mu"), power=power, Ntrial=Ntrial
            )
            sigma_derivative = [
                mc_sandwich(
                    d_omega,
                    variance_components.get("variance_sqrt_output"),
                    variance_components.get("variance_sqrt_output"),
                )
                for d_omega in z
            ]
            if not power_fixed:
                if variance in ["power", "binomialP"]:
                    sigma_derivative_power = np.array(
                        [
                            mc_sandwich_power(
                                omega,
                                variance_components.get("variance_sqrt_output"),
                                variance_components.get(
                                    "derivative_variance_sqrt_power"
                                ),
                            )
                            for omega in omegas
                        ]
                    )
                    sigma_derivative = [sigma_derivative_power, *sigma_derivative]
                elif variance == "binomialPQ":
                    sigma_derivative_p = mc_sandwich(
                        omegas,
                        variance_components.get("variance_sqrt_output"),
                        variance_components.get("derivative_variance_sqrt_p"),
                    )
                    sigma_derivative_q = mc_sandwich(
                        omegas,
                        variance_components.get("variance_sqrt_output"),
                        variance_components.get("derivative_variance_sqrt_q"),
                    )
                    sigma_derivative = [
                        sigma_derivative,
                        sigma_derivative_p,
                        sigma_derivative_q,
                    ]
        elif variance in ["poisson_tweedie", "geom_tweedie"]:
            variance_components = self.__generate_variance(
                variance_type="power", mu=mu.get("mu"), power=power, Ntrial=Ntrial
            )
            sigma_derivative = [
                mc_sandwich(
                    d_omega,
                    variance_components.get("variance_sqrt_output"),
                    variance_components.get("variance_sqrt_output"),
                )
                for d_omega in z
            ]
            if not power_fixed:
                sigma_derivative_power = mc_sandwich_power(
                    omegas,
                    variance_components.get("variance_sqrt_output"),
                    variance_components.get("derivative_variance_sqrt_power"),
                )
                sigma_derivative = [sigma_derivative_power, *sigma_derivative]
        return dict(sigma_derivative=sigma_derivative)

    def _sigma_between_values(self, rho, n_resp):
        def forcesymmetric(sigmab):
            sigmab_t = sigmab.transpose()
            sigmabsymmetric = np.matmul(sigmab_t, sigmab_t.transpose())
            diagonal_value = sigmabsymmetric.diagonal()
            if np.max(diagonal_value) == np.min(diagonal_value):
                return sigmabsymmetric
            else:
                sigmabsymmetric = sigmab
                sigmabsymmetric[np.triu_indices(n_resp, k=1)] = rho
                return sigmabsymmetric

        """sigma between method computes between for sequence calculations.

        It responds out with 2-position-tuple with sigma between and its derivative.
        """
        if n_resp == 1:
            return (1, 1)
        else:
            sigmab = diagonal(n_resp, np.full(n_resp, 1))
            sigmab[np.tril_indices(n_resp, k=-1)] = rho
            sigmab = forcesymmetric(sigmab)

            d_sigmab = self._mc_derivative_sigma_between(n_resp=n_resp)
            return (sigmab, d_sigmab)

    @lru_cache(maxsize=64)
    def _mc_derivative_sigma_between(self, n_resp):
        list_derivatives = list()
        position = list(combinations(range(n_resp), 2))
        n_par = int(n_resp * (n_resp - 1) / 2)
        for index in range(n_par):
            derivative = np.zeros((n_resp, n_resp))
            derivative[position[index][0], position[index][1]] = 1
            derivative[position[index][1], position[index][0]] = 1
            list_derivatives.append(derivative)

        return list_derivatives

    def __derivative_cholesky(self, d_sigma, inv_chol, chol):
        def faux(d_sigma, inv_chol, chol, inv_chol_transpose):
            csr_d_sigma_matrix = csr_matrix(d_sigma)
            csr_inv_chol_matrix = csr_matrix(inv_chol)
            csr_inv_chol_transpose_matrix = csr_matrix(inv_chol_transpose)

            matrix_operations = csr_inv_chol_matrix.dot(csr_d_sigma_matrix)
            matrix_operations = matrix_operations.dot(csr_inv_chol_transpose_matrix)

            matrix_operations = tril(matrix_operations)
            matrix_operations = matrix_operations.toarray()

            diagonal_indices = np.diag_indices_from(matrix_operations)
            matrix_operations[diagonal_indices] = (
                matrix_operations[diagonal_indices] / 2
            )
            return np.dot(chol, matrix_operations)

        cholesky_element = []
        inv_chol_transpose = inv_chol.transpose()
        for derivative in d_sigma:
            if not isinstance(derivative, list):
                cholesky_element.append(
                    faux(derivative, inv_chol, chol, inv_chol_transpose)
                )
            else:
                for der in derivative:
                    cholesky_element.append(
                        faux(der, inv_chol, chol, inv_chol_transpose)
                    )

        if isinstance(derivative, list):
            cholesky_element.reverse()
        return cholesky_element

    def __mc_transform_list_bdiag(self, list_mat, response_number, number_of_labels):
        def build_block_diag(value, response_number):
            block_value = block_diag(value)

            matrix_base = np.zeros(
                (
                    block_value.shape[0] * number_of_labels,
                    block_value.shape[1] * number_of_labels,
                )
            )

            lower = block_value.shape[0] * response_number
            upper = block_value.shape[1] + block_value.shape[1] * response_number

            matrix_base[lower:upper, lower:upper] = block_value

            return matrix_base

        output = [build_block_diag(d_chol, response_number) for d_chol in list_mat]
        return output

    def _generate_variance(
        self, variance_type: str, mu: np.array, power: int = 1, Ntrial: list = 1
    ):
        return self.__generate_variance(variance_type, mu, power, Ntrial)

    def __generate_variance(
        self, variance_type: str, mu: np.array, power: int = 1, Ntrial: list = 1
    ):
        if variance_type == "power":
            return self.__power_variance_attributes(mu, power)
        elif variance_type == "binomialP":
            return self.__binomialp_variance_attributes(mu, power, Ntrial)
        elif variance_type == "binomialPQ":
            return self.__binomialpq_variance_attributes(mu, power, Ntrial)

    def __power_variance_attributes(self, mu, power):
        mu_power = mu**power
        sqrt_mu_power = np.sqrt(mu_power)
        n_len = len(mu)

        variance_sqrt_output = diagonal(n=n_len, values=sqrt_mu_power)
        derivative_variance_sqrt_power = diagonal(
            n=n_len, values=((mu_power * np.log(mu)) / (2 * sqrt_mu_power))
        )
        derivative_variance_sqrt_mu = (mu ** (power - 1) * power) / (2 * sqrt_mu_power)

        return dict(
            variance_sqrt_output=variance_sqrt_output,
            derivative_variance_sqrt_power=derivative_variance_sqrt_power,
            derivative_variance_sqrt_mu=derivative_variance_sqrt_mu,
        )

    def __binomialp_variance_attributes(self, mu, power, ntrial):
        constant = 1 / ntrial
        mu_power = mu**power
        mu_power1 = (1 - mu) ** power
        mu1mu = constant * (mu_power * mu_power1)
        sqrt_mu1mu = np.sqrt(mu1mu)
        n_len = len(mu)

        variance_sqrt_output = diagonal(n=n_len, values=sqrt_mu1mu)
        derivative_variance_sqrt_power = diagonal(
            n=n_len,
            values=(np.log(1 - mu) * mu1mu + np.log(mu) * mu1mu) / (2 * sqrt_mu1mu),
        )
        derivative_variance_sqrt_mu = (
            constant * (mu_power1 * (mu ** (power - 1)) * power)
            - constant * (((1 - mu) ** (power - 1)) * mu_power * power)
        ) / (2 * sqrt_mu1mu)

        return dict(
            variance_sqrt_output=variance_sqrt_output,
            derivative_variance_sqrt_power=derivative_variance_sqrt_power,
            derivative_variance_sqrt_mu=derivative_variance_sqrt_mu,
        )

    def __binomialpq_variance_attributes(self, mu, power, ntrial):
        constant = 1 / ntrial
        p = power[0]
        q = power[1]

        mu_p = mu**p
        mu1_q = (1 - mu) ** q
        mu_p_mu_q = mu_p * mu1_q
        mu1mu = mu_p_mu_q * constant
        sqrt_mu1mu = np.sqrt(mu1mu)
        n_len = len(mu)

        denominator1 = 2 * sqrt_mu1mu
        denominator2 = denominator1 * ntrial

        variance_sqrt_output = diagonal(n=n_len, values=sqrt_mu1mu)

        derivative_variance_sqrt_p = diagonal(
            n=n_len, values=(mu_p_mu_q * np.log(mu)) / denominator2
        )

        derivative_variance_sqrt_q = diagonal(
            n=n_len, values=(mu_p_mu_q * np.log(1 - mu)) / denominator2
        )

        derivative_variance_sqrt_mu = (
            constant * (mu1_q * (mu ** (p - 1)) * p)
            - constant * ((1 - mu) ** (q - 1) * mu_p * q)
        ) / denominator1

        return dict(
            variance_sqrt_output=variance_sqrt_output,
            derivative_variance_sqrt_p=derivative_variance_sqrt_p,
            derivative_variance_sqrt_q=derivative_variance_sqrt_q,
            derivative_variance_sqrt_mu=derivative_variance_sqrt_mu,
        )

    def __parser_sigma(self, build_sigma_map):
        """parser method for _calculate_sigma output attributes
        Args:
            build_sigma_map (list): a list with dicts, with sigmas
        """
        list_sigmas = [
            [
                build_sigma_iteration.get("sigma_raw"),
                build_sigma_iteration.get("sigma_chol"),
                build_sigma_iteration.get("sigma_chol_inv"),
            ]
            for build_sigma_iteration in build_sigma_map
        ]

        sigma_raw = [sigmas[0] for sigmas in list_sigmas]
        sigma_chol = [sigmas[1] for sigmas in list_sigmas]
        sigma_chol_inv = [sigmas[2] for sigmas in list_sigmas]

        return (sigma_raw, sigma_chol, sigma_chol_inv)

    def __parser_sigma_derivatives(self, sigma_derivatives_map):
        list_sigmas = [
            [build_sigma_iteration.get("sigma_derivative")]
            for build_sigma_iteration in sigma_derivatives_map
        ]
        sigma_derivative = [sigmas[0] for sigmas in list_sigmas]
        return sigma_derivative

    def __parser_c_inverse(self, raw_c_components):

        sigma_chol_block_matrix = raw_c_components[0]
        sigma_chol_inv_block_matrix = raw_c_components[1]
        c_inverse = raw_c_components[2]

        return (sigma_chol_block_matrix, sigma_chol_inv_block_matrix, c_inverse)
