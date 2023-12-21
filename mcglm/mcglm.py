import numpy as np
import statsmodels.api as sm
import pandas as pd


from itertools import groupby
from scipy import stats
from scipy.stats import ncx2
from numpy.linalg import inv, slogdet
from scipy.linalg import block_diag
from statsmodels.iolib.summary import Summary
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod.families.links import probit, cauchy, CLogLog, LogLog
from .mcglmmean import MCGLMMean
from .mcglmvariance import MCGLMVariance
from .mcglmcattr import MCGLMCAttributes
from .utils import mc_sandwich


class MCGLM(MCGLMMean, MCGLMVariance):
    __doc__ = """
    MCGLM class that implements MCGLM stastical models. (Bonat, Jørgensen 2015)
        
    It extends GLM for multi-responses and dependent components by fitting second-moment assumptions.

    Parameters
    ----------
    endog : array_like
        1d array of endogenous response variable. In case of multiple responses, the user must pass the responses on a list.  
    exog : array_like
        A dataset with the endogenous matrix in a Numpy fashion. Since the library doesn't set an intercept by default, the user must add it. In the case of multiple responses, the user must pass the design matrices as a python list. 
    z : array_like
        List with matrices components of the linear covariance matrix.
    link : array_like, string or None
        Specification for the link function. The MCGLM library implements the following options: identity, logit, power, log, probit, cauchy, cloglog, loglog, negativebinomial. In the case of None, the library chooses the identity link. In multiple responses, user must pass values as list.  
    variance : array_like, string or None
        Specification for the variance function. The MCGLM library implements the following options: constant, tweedie, binomialP, binomialPQ, geom_tweedie, poisson_tweedie. In the case of None, the library chooses the constant link. In multiple responses, user must pass values as list.   
    offset : array_like or None
        Offset for continuous or count. In multiple responses, user must pass values as list.   
    Ntrial : array_like or None
        Ntrial for binomial responses. In multiple responses, user must pass values as list.
    power_fixed : array_like or None
        Parameter that allows estimation of power when variance functions is either tweedie, geom_tweedie or poisson_tweedie. In multiple responses, user must pass values as list. 
    maxiter : float or None
        Number max of iterations. Defaults to 200.
    tol : float or None
        Threshold of minimum absolute change on paramaters. Defaults to 0.0001.
    tuning : float or None
        Step size parameter. Defaults to 0.5.
    weights : array_like or None
        Weight matrix. Defaults to None.
        
    Examples
    ----------
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.scotland.load()
    >>> data.exog = sm.add_constant(data.exog)
    
    >>> model = sm.GLM(data.endog, data.exog, z=[mc_id(data.exog)],
    ...                      link="log", variance="tweedie",
    ...                      power=2, power_fixed=False)
    
    >>> model_results = model.fit()
    >>> model_results.mu
    >>> model_results.pearson_residuals
    >>> model_results.aic
    >>> model_results.bic
    >>> model_results.loglikelihood
    
    Notes
    -----
    MCGLM is a brand new model, which provides a solid statistical model for fitting multi-responses non-gaussian, dependent, or independent data based on second-moment assumptions. When a user instantiates an mcglm object, she must specify attributes such as link, variance, and z matrices; it will drive the overall behavior of the model.
    For more details, check articles and documentation provided.
    """

    def __init__(
        self,
        endog,
        exog,
        z,
        link=None,
        variance=None,
        offset=None,
        ntrial=None,
        power=None,
        power_fixed=None,
        maxiter=50,
        tol=0.001,
        tuning=1,
        weights=None,
    ):
        super(MCGLM, self).__init__()

        self._max_iter = maxiter
        self._tol = tol
        self._tuning = tuning
        self._weights = weights

        self._n_targets = None
        self._n_obs = None
        self._X = None
        self._y_values = None
        self._y_names = None
        self._z = None
        self._ntrial = None
        self._tau_initial = None
        self._beta_initial = None
        self._rho_initial = None
        self._offset = None
        self._link = None
        self._variance = None
        self._power_fixed = None
        self._power_initial = None

        (
            self._n_targets,
            self._n_obs,
            self._X,
            self._y_values,
            self._y_names,
            self._z,
            self._ntrial,
            self._tau_initial,
            self._beta_initial,
            self._rho_initial,
            self._offset,
            self._link,
            self._variance,
            self._power_fixed,
            self._power_initial,
        ) = self.__calculate_static_attributes(
            endog, exog, link, variance, z, offset, power, power_fixed, ntrial
        )

    @property
    def df_model(self):
        """Calculates the degree of freedom for the model."""
        return len(self._beta_initial[0]) + len(self._tau_initial[0])

    @property
    def df_resid(self):
        """Calculates the degree of freedom for the model residuals."""
        return self._n_obs - self.df_model

    def __calculate_static_attributes(
        self, endog, exog, link, variance, z, offset, power, power_fixed, ntrial
    ):
        """Base method to warm up the pivotal artifacts for model training. It fulfills the None with default values, sets the power value, and creates the base vectors.

        Args:
            endog (list or np.array): _description_
            exog (list or np.array): _description_
            link (list): _description_
            variance (list): _description_
            z (list): _description_
            offset (list): _description_
            power (list): _description_
            power_fixed (list): _description_
            ntrial (list): _description_
        """

        def initial_power(variance):
            if variance in [
                "binomialP",
                "tweedie",
                "geom_tweedie",
                "poisson_tweedie",
                "constant",
            ]:
                return 1
            elif variance == "binomialPQ":
                return (1, 1)
            else:
                return 0

        n_targets = len(exog) if isinstance(exog, list) else 1
        n_obs = exog[0].shape[0] if n_targets > 1 else exog.shape[0]

        set_offset = offset is None
        set_link = link is None
        set_variance = variance is None
        set_power_fixed = power_fixed is None
        set_power = power is None

        tau = list()
        beta = list()
        rho = list()

        offset_list = list()
        link_list = list()
        variance_list = list()
        power_fixed_list = list()
        power_list = list()
        y_values = list()
        y_names = list()

        if n_targets == 1:
            X = [self._reformat_dataset(exog)]
            y_values = [self._reformat_dataset(endog).values]
            y_names = [self._reformat_dataset(endog).name]
            z = [z]
            ntrial = [ntrial]
            tau = [[0 for i in range(len(z[0]))]]
            beta = [np.ones(X[0].shape[1])]
            rho = 0

            offset_list = [None] if set_offset else [offset]
            link_list = ["identity"] if set_link else [link]
            variance_list = ["constant"] if set_variance else [variance]
            power_fixed_list = [True] if set_power_fixed else [power_fixed]
            power_list = [initial_power(variance_list[0])] if set_power else [power]

        else:
            X = exog
            if not set_offset:
                offset_list = offset
            if not set_link:
                link_list = link
            if not set_variance:
                variance_list = variance
            if not set_power_fixed:
                power_fixed_list = power_fixed
            if not set_power:
                power_list = power

            for index in range(n_targets):
                tau.append([0 for i in range(len(z[index]))])
                beta.append([np.ones(exog[index].shape[1])])
                y_values.append(self._reformat_dataset(endog[index]).values)
                y_names.append(self._reformat_dataset(endog[index]).name)

                if set_offset:
                    offset_list.append(None)
                if set_link:
                    link_list.append("identity")
                if set_variance:
                    variance_list.append("constant")
                if set_power_fixed:
                    power_fixed_list.append(True)
                if set_power:
                    power_list.append(initial_power(variance[index]))

            rho = list()
            for _ in range(int(n_targets * (n_targets - 1) / 2)):
                rho.append(0)

        return (
            n_targets,
            n_obs,
            X,
            np.concatenate(y_values, axis=None),
            y_names,
            z,
            ntrial,
            tau,
            beta,
            rho,
            offset_list,
            link_list,
            variance_list,
            power_fixed_list,
            power_list,
        )
        
    def _reformat_dataset(self, dataset):
        if isinstance(dataset, np.ndarray):
            if len(dataset.shape) == 1:
                return pd.Series(dataset)
            else:
                return pd.DataFrame(dataset)    
        return dataset


    def fit(self):
        """The interface to run the inference for MCGLM statistical model."""
        (
            regression_historical,
            dispersion_historical,
            residue,
            varcov,
            joint_inv_sensitivity,
            joint_variability,
            n_iter,
            mu,
            rho,
            tau,
            power,
            parameters_target,
            c_inverse,
            c_values,
        ) = self._fit()

        regression_parameters = regression_historical[-1]
        dispersion_parameters = dispersion_historical[-1]

        p_log_likelihood = self.__p_log_likelihood(
            self._y_values, mu, c_values, c_inverse
        )
        aic = self.__p_aic(p_log_likelihood, self.df_model)
        bic = self.__p_bic(p_log_likelihood, self.df_model, len(self._y_values))

        return MCGLMResults(
            normalized_var_cov=abs(varcov),
            nobs=self._n_obs,
            n_targets=self._n_targets,
            y_names=self._y_names,
            regression=regression_parameters,
            dispersion=parameters_target,
            n_iter=n_iter,
            residue=residue,
            rho=rho,
            tau=tau,
            power=power,
            link=self._link,
            variance=self._variance,
            power_fixed=self._power_fixed,
            p_log_likelihood=p_log_likelihood,
            aic=aic,
            bic=bic,
            df_resid=self.df_resid,
            df_model=self.df_model,
            mu=mu.reshape(self._n_targets, -1),
            y_values=self._y_values,
            X=self._X,
            ntrial=self._ntrial,
        )

    def __p_log_likelihood(self, endog, mu, c_values, c_inverse):
        """
        Gaussian Pseudo-loglikelihood
        """

        def gauss(residue, det_sigma, inv_sigma):
            import math

            size_residue = len(residue)
            dens = (
                -1 * (size_residue / 2) * np.log(2 * math.pi)
                - 0.5 * det_sigma
                - (residue.T * inv_sigma * residue)
            )
            return dens.mean()

        residue = endog - mu
        det_sigma = slogdet(block_diag(c_values))[1]
        log_likelihood = gauss(
            residue=residue, det_sigma=det_sigma, inv_sigma=c_inverse
        )

        return log_likelihood

    def __p_aic(self, pseudo_loglike, degrees_of_freedom):
        """
        pseudo Akaike Information criterion
        """
        return round(2 * degrees_of_freedom - 2 * pseudo_loglike, 2)

    def __p_bic(self, pseudo_loglike, degrees_of_freedom, length_endogenous):
        """
        pseudo Bayesian Information criterion
        """
        return round(
            degrees_of_freedom * np.log(length_endogenous) - 2 * pseudo_loglike, 2
        )

    def _fit(self):
        """This method implements the core inference for MCGLM, by all the means of the two-moment assumptions."""
        W = (
            np.diag(np.ones(len(self._y_values)))
            if self._weights is None
            else self._weights
        )

        regression_historical = []
        dispersion_historical = []

        # To a warm up the start
        self.__update_optimal_dispersion()

        rho = self._rho_initial
        tau = self._tau_initial
        power = self._power_initial

        # Setting up
        regression = np.array(self._beta_initial)
        dispersion = np.array(self.__create_dispersion_vector(rho, power, tau))

        regression_historical.append(regression)
        dispersion_historical.append(dispersion)
        for iter in range(self._max_iter):
            # First moment

            (
                new_regression,
                quasi_score,
                mean_sensitivity,
                mean_variability,
            ) = self.update_beta(regression, W, power, rho, tau)
            regression_historical.append(new_regression)

            # Second moment
            mu_attributes, mu, mu_derivatives = self.calculate_mean_features(
                self._link, new_regression, self._X, self._offset
            )

            (
                new_dispersion,
                c_inverse,
                c_values,
                c_derivatives_componentes,
                var_sensitivity,
            ) = self.update_covariates(
                mu_attributes, rho, power, tau, W, dispersion, mu
            )

            dispersion_historical.append(new_dispersion)

            regression, dispersion, rho, power, tau = self.__iteration_update(
                new_regression, new_dispersion, rho, power, tau
            )
            if self.__check_stop_criterion(
                regression_historical, dispersion_historical
            ):
                break

        residue = self._y_values - mu

        (
            varcov,
            joint_inv_sensitivity,
            joint_variability,
        ) = MCGLMParameters._parameters_attributes(
            residue,
            mu_derivatives,
            W,
            quasi_score,
            mean_sensitivity,
            mean_variability,
            c_inverse,
            c_values,
            c_derivatives_componentes,
            var_sensitivity,
        )

        parameters_target = self.__get_dispersion_parameters(mu, rho, tau, dispersion)

        return (
            regression_historical,
            dispersion_historical,
            residue,
            varcov,
            joint_inv_sensitivity,
            joint_variability,
            str(iter),
            mu,
            rho,
            tau,
            power,
            parameters_target,
            c_inverse,
            c_values,
        )

    def __get_dispersion_parameters(self, mu, rho, tau, dispersion):

        covariances = dispersion.tolist()

        if self._n_targets == 1:
            rho = np.array([rho])
        elif self._n_targets > 1:
            rho = np.array([covariances.pop(0) for _ in range(len(rho))])

        parameters_target = list()
        for index in range(self._n_targets):
            power_mu = {
                "power": self._power_initial[index]
                if self._power_fixed[index]
                else covariances.pop(0),
                "mu": mu[index * self._n_obs : (index + 1) * self._n_obs],
            }

            size_parameters = (
                len(tau[index]) if self._power_fixed[index] else len(tau[index])
            )

            scale_output = {
                "scalelist": [
                    round(covariances.pop(0), 3) for _ in range(size_parameters)
                ]
            }

            parameters_target.append({**scale_output, **power_mu})

        return parameters_target

    def __check_stop_criterion(self, regression_historical, dispersion_historical):
        regression_lift = abs(regression_historical[-2] - regression_historical[-1])
        dispersion_lift = abs(dispersion_historical[-2] - dispersion_historical[-1])

        for position in range(len(regression_lift)):
            if np.all(regression_lift[position] < self._tol) and np.all(
                dispersion_lift < self._tol
            ):
                return True
        return False

    def __iteration_update(self, new_regression, new_dispersion, rho, power, tau):

        if self._n_targets == 1:
            if self._power_fixed[0]:
                return new_regression, new_dispersion, rho, power, [new_dispersion]
            else:
                return (
                    new_regression,
                    new_dispersion,
                    rho,
                    [new_dispersion[0]],
                    [new_dispersion[1:].tolist()],
                )

        if (isinstance(rho, list)) or (isinstance(rho, np.ndarray)):
            new_rho = new_dispersion[0 : len(rho)]
            stack_index = len(rho)
        else:
            new_rho = new_dispersion[0]
            stack_index = 1

        new_power = list()
        new_tau = list()
        for position in range(len(tau)):
            # power section
            if self._power_fixed[position]:
                new_power.append(power[position])
            else:
                new_power.append(new_dispersion[stack_index])
                stack_index += 1

            # tau section.
            additions = 0
            temp_tau = list()
            for index in range(len(tau[position])):
                temp_tau.append(new_dispersion[stack_index + index])
                additions += 1
            stack_index += additions
            new_tau.append(temp_tau)

        return new_regression, new_dispersion, new_rho, new_power, new_tau

    def __create_dispersion_vector(self, rho, power, tau):
        covariance = list()

        if self._n_targets > 1:
            if isinstance(rho, list):
                covariance.extend(rho)
            else:
                covariance.append(rho)
        for n_outputs in range(len(tau)):
            if not self._power_fixed[n_outputs]:
                covariance.append(power[n_outputs])
            covariance = covariance + tau[n_outputs]
        return covariance

    def __update_optimal_dispersion(self):
        """Update optimal dispersion method calculates optimal values for regression parameters and the dispersion. It harnesses the GLM API of statsmodels for calculating those values."""

        def logit_est(endog, exog, offset):
            """For the Logit link function, the method adjusts a GLM with Binomial family. It retrieves the parameters specified.

            Args:
                endog (array-like): outcome variable.
                exog (array-like): independent variables.
                offset (int): shift on response variable.

            Returns:
                tuple: regression parameters, dispersion parameter.
            """
            mdl = GLM(endog, exog, family=sm.families.Binomial(), offset=offset)
            mdl_results = mdl.fit()
            return mdl_results.params.values, mdl_results.scale

        def loglog_est(endog, exog, offset):
            """For the LogLog link function, the method adjusts a GLM with Binomial family. It retrieves the parameters specified.

            Args:
                endog (array-like): outcome variable.
                exog (array-like): independent variables.
                offset (int): shift on response variable.

            Returns:
                tuple: regression parameters, dispersion parameter.
            """
            mdl = GLM(endog, exog, family=sm.families.Binomial(LogLog()), offset=offset)
            mdl_results = mdl.fit()
            return mdl_results.params.values, mdl_results.scale

        def cloglog_est(endog, exog, offset):
            """For the CLogLog link function, the method adjusts a GLM with Binomial family. It retrieves the parameters specified.

            Args:
                endog (array-like): outcome variable.
                exog (array-like): independent variables.
                offset (int): shift on response variable.

            Returns:
                tuple: regression parameters, dispersion parameter.
            """
            mdl = GLM(
                endog, exog, family=sm.families.Binomial(CLogLog()), offset=offset
            )
            mdl_results = mdl.fit()
            return mdl_results.params.values, mdl_results.scale

        def cauchy_est(endog, exog, offset):
            """For the Cauchy link function, the method adjusts a GLM with Binomial family. It retrieves the parameters specified.

            Args:
                endog (array-like): outcome variable.
                exog (array-like): independent variables.
                offset (int): shift on response variable.

            Returns:
                tuple: regression parameters, dispersion parameter.
            """
            mdl = GLM(endog, exog, family=sm.families.Binomial(cauchy()), offset=offset)
            mdl_results = mdl.fit()
            return mdl_results.params.values, mdl_results.scale

        def probit_est(endog, exog, offset):
            """For the Cauchy link function, the method adjusts a GLM with Binomial family. It retrieves the parameters specified.

            Args:
                endog (array-like): outcome variable.
                exog (array-like): independent variables.
                offset (int): shift on response variable.

            Returns:
                tuple: regression parameters, dispersion parameter.
            """
            mdl = GLM(endog, exog, family=sm.families.Binomial(probit()), offset=offset)
            mdl_results = mdl.fit()
            return mdl_results.params.values, mdl_results.scale

        def identity_est(endog, exog, offset=None):
            """For the Identity link function, the method adjusts a OLS. It retrieves the parameters specified.

            Args:
                endog (array-like): outcome variable.
                exog (array-like): independent variables.
                offset (int): shift on response variable.

            Returns:
                tuple: regression parameters, dispersion parameter.
            """
            mdl = sm.OLS(endog, exog, offset=offset)
            mdl_results = mdl.fit()
            return mdl_results.params, mdl_results.scale

        def log_est(endog, exog, offset):
            """For the Log link function, the method adjusts a GLM with Tweedie(power=1) family. It retrieves the parameters specified.

            Args:
                endog (array-like): outcome variable.
                exog (array-like): independent variables.
                offset (int): shift on response variable.

            Returns:
                tuple: regression parameters, dispersion parameter.
            """
            mdl = GLM(
                endog, exog, family=sm.families.Tweedie(var_power=1), offset=offset
            )
            mdl_results = mdl.fit()
            return mdl_results.params, mdl_results.scale

        def power_est(endog, exog, offset):
            """For either the Power or Reciprocal link function, the method adjusts a GLM with Tweedie(power=2) family. It retrieves the parameters specified.

            Args:
                endog (array-like): outcome variable.
                exog (array-like): independent variables.
                offset (int): shift on response variable.

            Returns:
                tuple: regression parameters, dispersion parameter.
            """
            mdl = GLM(
                endog, exog, family=sm.families.Tweedie(var_power=2), offset=offset
            )
            mdl_results = mdl.fit()
            return mdl_results.params.values, mdl_results.scale

        def negative_binomial_est(endog, exog, offset):
            """For the Negative Binomial link function, the method adjusts a GLM with Tweedie(power=2) family. It retrieves the parameters specified.

            Args:
                endog (array-like): outcome variable.
                exog (array-like): independent variables.
                offset (int): shift on response variable.

            Returns:
                tuple: regression parameters, dispersion parameter.
            """
            mdl = GLM(
                endog, exog, family=sm.families.Tweedie(var_power=2), offset=offset
            )
            mdl_results = mdl.fit()
            return mdl_results.params.values, mdl_results.scale

        first_estimation = {
            "logit": logit_est,
            "identity": identity_est,
            "power": power_est,
            "log": log_est,
            "probit": probit_est,
            "cauchy": cauchy_est,
            "cloglog": cloglog_est,
            "loglog": loglog_est,
            "negativebinomial": negative_binomial_est,
            "inverse_power": log_est,
            "reciprocal": power_est,
        }

        for target in range(self._n_targets):

            self._beta_initial[target], dispersion = first_estimation[
                self._link[target]
            ](
                self._y_values[
                    target * self._n_obs : target * self._n_obs + self._n_obs
                ],
                self._X[target],
                self._offset[target],
            )
            self._tau_initial[target][0] = dispersion


class MCGLMParameters:
    """According to MCGLM specification, grounded for frequentist inference traits, the estimation of resulting parameters converge asymptotically to a gaussian distribution with tuple mean-variance = (actual parameters, inverse of matrix Godambe). This property allows the calculation of pivotal traits regarding the parameters, such as: hypothesis testing and confidence interval.

    This class implements every method related to this trait.
    """

    @staticmethod
    def _parameters_attributes(
        resid,
        mu_deriv,
        W,
        quasi_score,
        mean_sens,
        mean_varia,
        c_inv,
        c,
        c_deriv,
        var_sensi,
    ):
        """The parameters of MCGLM converge assymtoptically to a Normal Distribution. This trait allows some statistical inferences as hypothesis testing, confidence interval and so on. This method crafts three important matrices: Varcov: variance covariance matrix, joint_inv_sensitivity: inverse sensitivity, and joint_variability: the joint variability distribution."""

        variance_variability = MCGLMParameters.generate_var_variability(
            resid, W, c_inv, c, c_deriv
        )

        inv_cw = np.dot(c_inv, W)

        s_cov_beta = MCGLMParameters._mc_cross_sensitivity(
            cov_product=c_deriv,
            columns_size=len(quasi_score),
        )

        v_cov_beta = MCGLMParameters._mc_cross_variability(
            cov_product=c_deriv,
            inv_cw=inv_cw,
            res=resid,
            d=mu_deriv,
        )

        p1 = np.append(mean_varia, v_cov_beta.transpose(), axis=0)

        p2 = np.append(v_cov_beta, variance_variability, axis=0)
        joint_variability = np.append(p1, p2, axis=1)

        inv_J_beta = inv(mean_sens)
        inv_S_beta = inv_J_beta
        inv_S_cov = inv(var_sensi)
        mat0 = np.zeros((s_cov_beta.shape[1], s_cov_beta.shape[0]))

        cross_term = mc_sandwich(s_cov_beta, -inv_S_cov, inv_S_beta)
        p1 = np.append(inv_S_beta, cross_term, axis=0)
        p2 = np.append(mat0, inv_S_cov, axis=0)

        joint_inv_sensitivity = np.append(p1, p2, axis=1)
        varcov = mc_sandwich(
            joint_variability, joint_inv_sensitivity, joint_inv_sensitivity.transpose()
        )

        return (
            varcov,
            joint_inv_sensitivity,
            joint_variability,
        )

    def _mc_cross_sensitivity(cov_product, columns_size):
        nrow = len(cov_product)
        return np.zeros((nrow, columns_size))

    def generate_var_variability(res, w, c_inv, c_val, c_comp):

        return MCGLMParameters._calculate_variability(
            product=c_comp,
            inv_C=c_inv,
            C=c_val,
            res=res,
            W=w,
        )

    def _calculate_variability(product, inv_C, C, res, W):
        n_par = len(product)
        we = [np.dot(product[index], inv_C) for index in range(n_par)]

        k4 = res**4 - 3 * np.diag(C) ** 2
        sensitivity = MCGLMVariance.generate_sensitivity(product, W=W**2)
        W = np.diag(W).flatten()

        variability = MCGLMParameters._mc_variability(sensitivity, we, k4, W)

        return variability

    def _mc_variability(sensitivity, we, k4, w):
        variability = np.array([])
        for position_row in range(len(we)):
            wi = np.diag(we[position_row])
            for position_col in range(len(we)):
                wj = np.diag(we[position_col])

                k4_operation = np.sum(k4 * wi * w * wj * w)

                variability = np.append(
                    variability,
                    -2 * sensitivity.item((position_row, position_col)) + k4_operation,
                )

        return variability.reshape((len(we), len(we)))

    def _covprod(a, w, res):
        calculation_sandwich = np.dot(np.dot(res, w), res)
        calculation_residue = np.dot(res.transpose(), a)
        product = np.dot(calculation_sandwich, calculation_residue)
        return product

    def _mc_cross_variability(cov_product, inv_cw, res, d):

        wlist = [np.dot(cov, inv_cw) for cov in cov_product]
        a = np.dot(d.transpose(), inv_cw)

        n_beta = a.shape[0]
        n_cov = len(cov_product)
        cross_variability = []
        for cov in range(n_cov):
            for beta in range(n_beta):
                cross_variability.append(
                    MCGLMParameters._covprod(a[beta, :], wlist[cov], res)
                )
        cross_variability = np.array(cross_variability).reshape(n_cov, n_beta).T
        return cross_variability


class MCGLMResults(GLMResults):
    """MCGLM Class for generating and manipulating results of mcglm training. The main output goes by the method summary(), the classical statsmodels output. Therefore, the user can access the attributes "aic", "bic" e loglikelihood.

    Args:
        GLMResults: Class of statsmodels library for presenting results of GLM.
    """

    def __init__(
        self,
        normalized_var_cov,
        nobs,
        n_targets,
        y_names,
        regression,
        dispersion,
        n_iter,
        residue,
        rho,
        tau,
        power,
        link,
        variance,
        power_fixed,
        p_log_likelihood,
        aic,
        bic,
        df_resid,
        df_model,
        mu,
        y_values,
        X,
        ntrial,
    ):
        self._normalized_var_cov = normalized_var_cov
        self._nobs = nobs
        self._n_targets = n_targets
        self._y_names = y_names
        self._regression = regression
        self._dispersion = dispersion
        self._n_iter = n_iter
        self._residue = residue
        self._rho = rho
        self._tau = tau
        self._power_list = power
        self._link_list = link
        self._variance_list = variance
        self._power_fixed_list = power_fixed
        self._p_log_likelihood = p_log_likelihood
        self._p_aic = aic
        self._p_bic = bic
        self._df_resid = df_resid
        self._df_model = df_model
        self._mu_value = mu
        self._y_values = y_values
        self._X = X
        self._ntrial = ntrial
        self.params = None
        # self.bse = None

        self._use_t = False
        self.model = None

    @property
    def aic(self):
        return self._p_aic

    @property
    def bic(self):
        return self._p_bic

    @property
    def loglikelihood(self):
        return self._p_log_likelihood

    @property
    def bse(self):
        """The standard errors of the parameter estimates."""
        if (not hasattr(self, "cov_params_default")) and (
            self.normalized_cov_params is None
        ):
            bse_ = np.empty(len(self.params))
            bse_[:] = np.nan
        else:
            bse_ = np.sqrt(np.diag(self.normalized_cov_params))
        return bse_

    @property
    def tvalues(self):
        """
        Return the t-statistic for a given parameter estimate.
        """
        return self.params / self.bse

    @property
    def mu(self):
        return self._mu_value

    @property
    def pvalues(self):
        """The two-tailed p values for the t-stats of the params."""
        if self.use_t:
            df_resid = getattr(self, "df_resid_inference", self._df_resid)
            return stats.t.sf(np.abs(self.tvalues), df_resid) * 2
        else:
            return stats.norm.sf(np.abs(self.tvalues)) * 2

    @property
    def pearson_residuals(self):
        """
        Pearson residuals.  The Pearson residuals are defined as
        (`endog` - `mu`)/sqrt(VAR(`mu`)) where VAR is the distribution
        specific variance function.  See statsmodels.families.family and
        statsmodels.families.varfuncs for more information.
        """
        residuals = []
        if self._n_targets == 1:
            mu = [self.mu]
        else:
            mu = self.mu.reshape(self._n_targets, -1)

        for index in range(self._n_targets):
            residue = (
                self._y_values[index * self._nobs : index * self._nobs + self._nobs]
                - mu[index]
            )

            variance = None
            if self._variance_list[index] in ("binomialP", "binomialPQ", "constant"):
                variance = self._variance_list[index]
            else:
                variance = "power"

            variance_sqrt_output = (
                MCGLMCAttributes()
                ._generate_variance(
                    variance,
                    mu[index],
                    self._power_list[index],
                    self._ntrial[index],
                )
                .get("variance_sqrt_output")
            )

            residuals.append(residue / np.diag(variance_sqrt_output))

        return residuals

    @property
    def vcov(self):
        return self.normalized_cov_params

    def anova(
        self, indexes_covariates=[[1, 2, 2, 2, 2]], covariate_name=[["x1", "x2"]]
    ):

        total_anovas = list()
        for position_target in range(self.model.n_targets):
            dispersion = self.model.dispersion[position_target]["scalelist"][1:].copy()
            dispersion_vcov = self._dispersion_vcov[position_target][1:, 1:].copy()

            covariates_positions = [
                list(j) for i, j in groupby(indexes_covariates[position_target])
            ]
            covariates_names = covariate_name[position_target]

            index = 0
            index_name = 0
            anovas = list()
            for covs in covariates_positions:

                current_cov = np.array(dispersion[index : index + len(covs)])
                current_vcov = dispersion_vcov[
                    index : index + len(covs), index : index + len(covs)
                ]

                try:
                    solve_vcoc = inv(current_vcov)
                except Exception as e:
                    solve_vcoc = current_vcov

                chi_square = np.dot(
                    current_cov.transpose(), np.dot(solve_vcoc, current_cov)
                )
                df = len(covs)
                pvalue = ncx2.pdf(chi_square, df, nc=0)

                index += len(covs)
                anovas.append(
                    {
                        "covariate_name": covariates_names[index_name],
                        "chi_square": round(chi_square, 4),
                        "df": df,
                        "pvalue": round(pvalue, 4),
                    }
                )

                index_name += 1
            total_anovas.append(anovas)

        return total_anovas

    def __add_table_two_columns(
        self, summ, yname="output", xname=None, title=None, alpha=0.05
    ):
        top_left = [
            ("Dep. Variable:", None),
            ("Model:", ["MCGLM"]),
            ("link:", [self.link]),
            ("variance:", [self.variance]),
            ("Method:", ["Quasi-Likelihood"]),
            ("Date:", None),
            ("Time:", None),
        ]

        top_right = [
            ("No. Iterations:", [self._n_iter]),
            ("No. Observations:", [self._nobs]),
            ("Df Residuals:", [self._df_resid]),
            ("Df Model:", [self._df_model]),
            ("Power-fixed:", [self.power_fixed]),
            ("pAIC", [self.paic]),
            ("pBIC", [self.pbic]),
            ("pLogLik", [round(self._p_log_likelihood, 4)]),
        ]

        if hasattr(self, "cov_type"):
            top_left.append(("Covariance Type:", [self.cov_type]))

        if title is None:
            title = "Multivariate Covariance Generalized Linear Model"

        summ.add_table_2cols(
            self,
            gleft=top_left,
            gright=top_right,
            yname=yname,
            xname=xname,
            title=title,
        )
        summ.add_table_params(
            self, yname=yname, xname=xname, alpha=alpha, use_t=self.use_t
        )

        return summ

    def __add_dispersion(self, summ, title=None, alpha=0.05):
        summ.add_table_params(
            self,
            alpha=alpha,
            xname=["dispersion_" + str(i + 1) for i in range(len(self.params))],
            use_t=self.use_t,
        )

        return summ

    def __add_power(self, summ, alpha=0.05):
        summ.add_table_params(
            self,
            alpha=alpha,
            xname=[
                "power_" + (str(i + 1) if len(self.params) > 1 else "")
                for i in range(len(self.params))
            ],
            use_t=self.use_t,
        )

        return summ

    def __add_rho_section(self, summ, alpha=0.05):
        self.normalized_cov_params = self._rho_vcov
        self.params = np.array(self._rho) if isinstance(self._rho, int) else self._rho
        if isinstance(self._rho, (np.ndarray, np.generic)):
            summ.add_table_params(
                self,
                alpha=alpha,
                xname=[
                    "rho_" + (str(i + 1) if len(self.params) > 1 else "")
                    for i in range(len(self.params))
                ],
                use_t=self.use_t,
            )

            if hasattr(self, "constraints"):
                summ.add_extra_txt(
                    [
                        "Model has been estimated subject to linear "
                        "equality constraints."
                    ]
                )
        return summ

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """
        It generates the summary report as the sketch of classical "statsmodels" library. The summary shows all parameters found thoroughly, for each response.
        """
        self._dispersion_vcov = []
        self.betas_vcov = []
        self._rho_vcov = []
        self.power_vcov = []

        index_position = 0
        cov_params = self._normalized_var_cov

        if self._n_targets == 1:
            self._rho_vcov = np.array([np.nan])
            mu = [self.mu]
        else:
            mu = self.mu

        for index in range(self._n_targets):

            n_betas = len(self._regression[index])
            self.betas_vcov.append(
                cov_params[
                    index_position : index_position + n_betas,
                    index_position : index_position + n_betas,
                ]
            )
            index_position += n_betas

        if self._n_targets > 1:
            self._rho_vcov = cov_params[
                index_position : index_position + len(self._rho),
                index_position : index_position + len(self._rho),
            ]

            index_position += len(self._rho)

        for index in range(self._n_targets):

            if self._power_fixed_list[index]:
                self.power_vcov.append(np.array([np.nan]))
            else:
                self.power_vcov.append(
                    np.array(
                        cov_params[
                            index_position : index_position + 1,
                            index_position : index_position + 1,
                        ]
                    )
                )
                index_position += 1

            n_dispersion = len(self._dispersion[index]["scalelist"])

            self._dispersion_vcov.append(
                cov_params[
                    index_position : index_position + n_dispersion,
                    index_position : index_position + n_dispersion,
                ]
            )

            index_position += n_dispersion

        self.scale = 1

        smry = Summary()

        for target_index in range(self._n_targets):
            exog_names = self._X[target_index].columns.tolist()
            y_name = self._y_names[target_index]

            self.link = self._link_list[target_index]
            self.variance = self._variance_list[target_index]
            self.power_fixed = self._power_fixed_list[target_index]
            self.params = self._regression[target_index]
            self.normalized_cov_params = self.betas_vcov[target_index]

            self.loglikehood = self._p_log_likelihood
            self.paic = self._p_aic
            self.pbic = self._p_bic

            smry = self.__add_table_two_columns(smry, yname=y_name, xname=exog_names)

            self.params = np.array(self._dispersion[target_index]["scalelist"])
            self.normalized_cov_params = self._dispersion_vcov[target_index]

            # self.bse = np.sqrt(np.diag(self.normalized_cov_params))
            smry = self.__add_dispersion(smry)
            self.params = np.array([self._dispersion[target_index]["power"]])
            self.normalized_cov_params = self.power_vcov[target_index]

            smry = self.__add_power(smry)

        smry = self.__add_rho_section(smry)

        return smry
