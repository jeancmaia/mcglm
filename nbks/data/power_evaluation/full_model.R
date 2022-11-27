fit_mcglm <- function(list_initial, list_link, list_variance,
                      list_covariance, list_X, list_Z,
                      list_offset, list_Ntrial, list_power_fixed,
                      list_sparse, y_vec,
                      correct = FALSE, max_iter, tol = 0.001,
                      method = "rc",
                      tuning = 0, verbose, weights) {
    ## Diagonal matrix with weights
    W <- Diagonal(length(y_vec), weights)
    ## Transformation from list to vector
    parametros <- mc_list2vec(list_initial, list_power_fixed)
    n_resp <- length(list_initial$regression)
    if (n_resp == 1) {
        parametros$cov_ini <- parametros$cov_ini[-1]
    }
    ## Getting information about the number of parameters
    inf <- mc_getInformation(list_initial, list_power_fixed, n_resp = n_resp)
    ## Creating a matrix to sote all values used in the fitting step
    solucao_beta <- matrix(NA, max_iter, length(parametros$beta_ini))
    solucao_cov <- matrix(NA, max_iter, length(parametros$cov_ini))
    score_beta_temp <- matrix(NA, max_iter, length(parametros$beta_ini))
    score_disp_temp <- matrix(NA, max_iter, length(parametros$cov_ini))
    ## Setting the initial values
    solucao_beta[1, ] <- parametros$beta_ini
    solucao_cov[1, ] <- parametros$cov_ini
    beta_ini <- parametros$beta_ini
    cov_ini <- parametros$cov_ini
    for (i in 2:max_iter) {
        print('dispersion_current')
        print(cov_ini)
        ## Step 1 - Quasi-score function Step 1.1 - Computing the mean structure
        mu_list <- Map(mc_link_function, beta = list_initial$regression,
                       offset = list_offset, X = list_X, link = list_link)
        mu_vec <- do.call(c, lapply(mu_list, function(x) x$mu))
        D <- bdiag(lapply(mu_list, function(x) x$D))
        # Step 1.2 - Computing the inverse of C matrix.
        # I should improve this step.
        # I have to write a new function to compute
        # only C or inv_C to be more efficient in this step.
        Cfeatures <- mc_build_C(list_mu = mu_list,
                                list_Ntrial = list_Ntrial,
                                rho = list_initial$rho,
                                list_tau = list_initial$tau,
            list_power = list_initial$power, list_Z = list_Z,
            list_sparse = list_sparse, list_variance = list_variance,
            list_covariance = list_covariance,
            list_power_fixed = list_power_fixed, compute_C = FALSE,
            compute_derivative_beta = FALSE,
            compute_derivative_cov = FALSE)
        # Step 1.3 - Update the regression parameters
        beta_temp <- mc_quasi_score(D = D, inv_C = Cfeatures$inv_C,
                                    y_vec = y_vec, mu_vec = mu_vec,
                                    W = W)
        solucao_beta[i, ] <- as.numeric(beta_ini - solve(beta_temp$Sensitivity, beta_temp$Score))
        score_beta_temp[i, ] <- as.numeric(beta_temp$Score)
        list_initial <- mc_updateBeta(list_initial, solucao_beta[i, ],
                                      information = inf, n_resp = n_resp)
        # Step 1.4 - Updated the mean structure to use in the Pearson
        # estimating function step.
        mu_list <- Map(mc_link_function, beta = list_initial$regression,
                       offset = list_offset, X = list_X, link = list_link)
        mu_vec <- do.call(c, lapply(mu_list, function(x) x$mu))
        D <- bdiag(lapply(mu_list, function(x) x$D))
        # Step 2 - Updating the covariance parameters
        Cfeatures <- mc_build_C(list_mu = mu_list,
                                list_Ntrial = list_Ntrial,
                                rho = list_initial$rho,
                                list_tau = list_initial$tau,
            list_power = list_initial$power, list_Z = list_Z,
            list_sparse = list_sparse, list_variance = list_variance,
            list_covariance = list_covariance,
            list_power_fixed = list_power_fixed, compute_C = TRUE,
            compute_derivative_beta = FALSE)
        print('c_inverse')
        print(sum(Cfeatures$inv_C))
        print('c_derivative')
        print(sum(Cfeatures$D_C[[1]]))
        print(sum(Cfeatures$D_C[[2]]))
        # Step 2.1 - Using beta(i+1)
        #beta_temp2 <- mc_quasi_score(D = D, inv_C = Cfeatures$inv_C,
        #y_vec = y_vec, mu_vec =  mu_vec)
        inv_J_beta <- solve(beta_temp$Sensitivity)
        if (method == "chaser") {
            cov_temp <- mc_pearson(y_vec = y_vec, mu_vec = mu_vec,
                                   Cfeatures = Cfeatures,
                                   inv_J_beta = inv_J_beta, D = D,
                                   correct = correct,
                                   compute_variability = FALSE,
                                   W = W)
            print('pearson_score')
            print(sum(cov_temp$Score))
            print('sensitivity')
            print(sum(cov_temp$Sensitivity))
            step <- tuning * solve(cov_temp$Sensitivity, cov_temp$Score)
        }
        if(method == "gradient") {
        cov_temp <- mc_pearson(y_vec = y_vec, mu_vec = mu_vec,
                                 Cfeatures = Cfeatures,
                                 inv_J_beta = inv_J_beta, D = D,
                                 correct = correct,
                                 compute_sensitivity = FALSE,
                                 compute_variability = FALSE, W = W)
          step <- tuning * cov_temp$Score
        }
        if (method == "rc") {
            cov_temp <- mc_pearson(y_vec = y_vec, mu_vec = mu_vec,
                                   Cfeatures = Cfeatures,
                                   inv_J_beta = inv_J_beta, D = D,
                                   correct = correct,
                                   compute_variability = TRUE, W = W)
            step <- solve(tuning * cov_temp$Score %*% t(cov_temp$Score)
                          %*% solve(cov_temp$Variability) %*%
                            cov_temp$Sensitivity + cov_temp$Sensitivity)%*% cov_temp$Score
        }
        ## Step 2.2 - Updating the covariance parameters
        score_disp_temp[i, ] <- cov_temp$Score
        cov_next <- as.numeric(cov_ini - step)
        list_initial <- mc_updateCov(list_initial = list_initial,
                                     list_power_fixed = list_power_fixed,
                                     covariance = cov_next,
                                     information = inf, n_resp = n_resp)
        ## print the parameters values
        if (verbose == TRUE) {
            print(round(cov_next, 4))
        }
        if (verbose == TRUE) {
            print(round(as.numeric(cov_temp$Score), 4))
        }
        ## Step 2.3 - Updating the initial values for the next step
        beta_ini <- solucao_beta[i, ]
        cov_ini <- cov_next
        solucao_cov[i, ] <- cov_next
        print('new_dispersion')
        print(cov_next)
        ## Checking the convergence
        #sol = abs(c(solucao_beta[i, ], solucao_cov[i, ]))
        tolera <- abs(c(solucao_beta[i, ], solucao_cov[i, ]) - c(solucao_beta[i - 1, ], solucao_cov[i - 1, ]))
        #tolera <- tolera/sol
        # if(verbose == TRUE){print(round(tolera, 4))}
        if (all(tolera <= tol) == TRUE)
            break
    }
    mu_list <- Map(mc_link_function, beta = list_initial$regression,
                   offset = list_offset, X = list_X, link = list_link)
    mu_vec <- do.call(c, lapply(mu_list, function(x) x$mu))
    D <- bdiag(lapply(mu_list, function(x) x$D))
    Cfeatures <- mc_build_C(list_mu = mu_list, list_Ntrial = list_Ntrial,
                            rho = list_initial$rho,
                            list_tau = list_initial$tau,
                            list_power = list_initial$power,
                            list_Z = list_Z, list_sparse = list_sparse,
                            list_variance = list_variance,
                            list_covariance = list_covariance,
                            list_power_fixed = list_power_fixed,
                            compute_C = TRUE,
                            compute_derivative_beta = FALSE)
    beta_temp2 <- mc_quasi_score(D = D, inv_C = Cfeatures$inv_C,
                                 y_vec = y_vec, mu_vec = mu_vec, W = W)
    inv_J_beta <- solve(beta_temp2$Sensitivity)

    cov_temp <- mc_pearson(y_vec = y_vec, mu_vec = mu_vec,
                           Cfeatures = Cfeatures, inv_J_beta = inv_J_beta,
                           D = D, correct = correct,
                           compute_variability = TRUE, W = W)
    #### Here I need to compute the cross-sensitivity and variability
    inv_CW <- Cfeatures$inv_C%*%W
    Product_beta <- lapply(Cfeatures$D_C_beta, mc_multiply,
                           bord2 = inv_CW)
    S_cov_beta <- mc_cross_sensitivity(Product_cov = cov_temp$Extra,
                                       Product_beta = Product_beta,
                                       n_beta_effective = length(beta_temp$Score))
    res <- y_vec - mu_vec
    V_cov_beta <- mc_cross_variability(Product_cov = cov_temp$Extra,
                                       inv_C = inv_CW, res = res, D = D)
    p1 <- rbind(beta_temp2$Variability, t(V_cov_beta))
    p2 <- rbind(V_cov_beta, cov_temp$Variability)
    joint_variability <- cbind(p1, p2)
    inv_S_beta <- inv_J_beta
    # Problem 1 x 1 matrix
    # IMPROVE IT
    inv_S_cov <- solve(as.matrix(cov_temp$Sensitivity))
    mat0 <- Matrix(0, ncol = dim(S_cov_beta)[1], nrow = dim(S_cov_beta)[2])
    cross_term <- -inv_S_cov %*% S_cov_beta %*% inv_S_beta
    p1 <- rbind(inv_S_beta, cross_term)
    p2 <- rbind(mat0, inv_S_cov)
    joint_inv_sensitivity <- cbind(p1, p2)
    VarCov <- joint_inv_sensitivity %*% joint_variability %*% t(joint_inv_sensitivity)
    output <- list(IterationRegression = solucao_beta,
                   IterationCovariance = solucao_cov,
                   ScoreRegression = score_beta_temp,
                   ScoreCovariance = score_disp_temp,
                   Regression = beta_ini,
                   Covariance = cov_ini,
                   vcov = VarCov, fitted = mu_vec,
                   residuals = res, inv_C = Cfeatures$inv_C,
                   C = Cfeatures$C, Information = inf,
                   mu_list = mu_list, inv_S_beta = inv_S_beta,
                   joint_inv_sensitivity = joint_inv_sensitivity,
                   joint_variability = joint_variability,
                   W = W)
    return(output)
}



mc_build_sigma <- function(mu, Ntrial = 1, tau, power, Z, sparse,
                           variance, covariance, power_fixed,
                           compute_derivative_beta = FALSE) {
    if (variance == "constant") {
        if (covariance == "identity" | covariance == "expm") {
            Omega <- mc_build_omega(tau = tau, Z = Z,
                                    covariance_link = covariance,
                                    sparse = sparse)
            chol_Sigma <- chol(Omega$Omega)
            inv_chol_Sigma <- solve(chol_Sigma)
            output <- list(Sigma_chol = chol_Sigma,
                           Sigma_chol_inv = inv_chol_Sigma,
                           D_Sigma = Omega$D_Omega)
        }
        if (covariance == "inverse") {
            inv_Sigma <- mc_build_omega(tau = tau, Z = Z,
                                        covariance_link = "inverse",
                                        sparse = sparse)
            chol_inv_Sigma <- chol(inv_Sigma$inv_Omega)
            chol_Sigma <- solve(chol_inv_Sigma)
            ## Because a compute the inverse of chol_inv_Omega
            Sigma <- (chol_Sigma) %*% t(chol_Sigma)
            D_Sigma <- lapply(inv_Sigma$D_inv_Omega,
                           mc_sandwich_negative,
                              bord1 = Sigma, bord2 = Sigma)
            output <- list(Sigma_chol = t(chol_Sigma),
                           Sigma_chol_inv = t(chol_inv_Sigma),
                           D_Sigma = D_Sigma)
        }
    }

    if (variance == "tweedie" | variance == "binomialP" |
            variance == "binomialPQ") {
        if (variance == "tweedie") {
            variance <- "power"
        }
        if (covariance == "identity" | covariance == "expm") {
            Omega <- mc_build_omega(tau = tau, Z = Z,
                                    covariance_link = covariance,
                                    sparse = sparse)
            V_sqrt <- mc_variance_function(
                mu = mu$mu, power = power,
                Ntrial = Ntrial,
                variance = variance,
                inverse = FALSE,
                derivative_power = !power_fixed,
                derivative_mu = compute_derivative_beta)
            Sigma <- forceSymmetric(V_sqrt$V_sqrt %*% Omega$Omega %*%
                                        V_sqrt$V_sqrt)
            chol_Sigma <- chol(Sigma)
            inv_chol_Sigma <- solve(chol_Sigma)
            D_Sigma <- lapply(Omega$D_Omega, mc_sandwich,
                              bord1 = V_sqrt$V_sqrt,
                              bord2 = V_sqrt$V_sqrt)
            if (power_fixed == FALSE) {
                if (variance == "power" | variance == "binomialP") {
                    print('omegas')
                    print(sum(Omega$Omega))
                    print('variance_sqrt_output')
                    print(sum(V_sqrt$V_sqrt))
                    print('derivative_variance_sqrt_power')
                    print(sum(V_sqrt$D_V_sqrt_p))
                    D_Sigma_power <- mc_sandwich_power(
                        middle = Omega$Omega, bord1 = V_sqrt$V_sqrt,
                        bord2 = V_sqrt$D_V_sqrt_p)
                    print("D_sigma_power")
                    print(sum(D_Sigma_power))
                    D_Sigma <- c(D_Sigma_power = D_Sigma_power,
                                 D_Sigma_tau = D_Sigma)
                }
                if (variance == "binomialPQ") {
                    D_Sigma_p <- mc_sandwich_power(
                        middle = Omega$Omega,
                        bord1 = V_sqrt$V_sqrt,
                        bord2 = V_sqrt$D_V_sqrt_p)
                    D_Sigma_q <- mc_sandwich_power(
                        middle = Omega$Omega,
                        bord1 = V_sqrt$V_sqrt,
                        bord2 = V_sqrt$D_V_sqrt_q)
                    D_Sigma <- c(D_Sigma_p, D_Sigma_q, D_Sigma)
                }
            }
            output <- list(Sigma_chol = chol_Sigma,
                           Sigma_chol_inv = inv_chol_Sigma,
                           D_Sigma = D_Sigma)
            if (compute_derivative_beta == TRUE) {
                D_Sigma_beta <- mc_derivative_sigma_beta(
                    D = mu$D, D_V_sqrt_mu = V_sqrt$D_V_sqrt_mu,
                    Omega = Omega$Omega, V_sqrt = V_sqrt$V_sqrt,
                    variance = variance)
                output$D_Sigma_beta <- D_Sigma_beta
            }
        }
        if (covariance == "inverse") {
            inv_Omega <- mc_build_omega(tau = tau, Z = Z,
                                        covariance_link = "inverse",
                                        sparse = sparse)
            V_inv_sqrt <- mc_variance_function(
                mu = mu$mu, power = power, Ntrial = Ntrial,
                variance = variance, inverse = TRUE,
                derivative_power = !power_fixed,
                derivative_mu = compute_derivative_beta)
            inv_Sigma <- forceSymmetric(V_inv_sqrt$V_inv_sqrt %*%
                                            inv_Omega$inv_Omega %*%
                                            V_inv_sqrt$V_inv_sqrt)
            inv_chol_Sigma <- chol(inv_Sigma)
            chol_Sigma <- solve(inv_chol_Sigma)
            Sigma <- chol_Sigma %*% t(chol_Sigma)
            D_inv_Sigma <- lapply(inv_Omega$D_inv_Omega, mc_sandwich,
                                  bord1 = V_inv_sqrt$V_inv_sqrt,
                                  bord2 = V_inv_sqrt$V_inv_sqrt)
            D_Sigma <- lapply(D_inv_Sigma, mc_sandwich_negative,
                              bord1 = Sigma, bord2 = Sigma)
            if (power_fixed == FALSE) {
                if (variance == "power" | variance == "binomialP") {
                    D_Omega_p <- mc_sandwich_power(
                        middle = inv_Omega$inv_Omega,
                        bord1 = V_inv_sqrt$V_inv_sqrt,
                        bord2 = V_inv_sqrt$D_V_inv_sqrt_power)
                    D_Sigma_p <- mc_sandwich_negative(
                        middle = D_Omega_p,
                        bord1 = Sigma, bord2 = Sigma)
                    D_Sigma <- c(D_Sigma_p, D_Sigma)
                }
                if (variance == "binomialPQ") {
                    D_Omega_p <- mc_sandwich_power(
                        middle = inv_Omega$inv_Omega,
                        bord1 = V_inv_sqrt$V_inv_sqrt,
                        bord2 = V_inv_sqrt$D_V_inv_sqrt_p)
                    D_Sigma_p <- mc_sandwich_negative(
                        middle = D_Omega_p,
                        bord1 = Sigma, bord2 = Sigma)
                    D_Omega_q <- mc_sandwich_power(
                        middle = inv_Omega$inv_Omega,
                        bord1 = V_inv_sqrt$V_inv_sqrt,
                        bord2 = V_inv_sqrt$D_V_inv_sqrt_q)
                    D_Sigma_q <- mc_sandwich_negative(
                        middle = D_Omega_q,
                        bord1 = Sigma, bord2 = Sigma)
                    D_Sigma <- c(D_Sigma_p, D_Sigma_q, D_Sigma)
                }
            }
            output <- list(Sigma_chol = t(chol_Sigma),
                           Sigma_chol_inv = t(inv_chol_Sigma),
                           D_Sigma = D_Sigma)
            if (compute_derivative_beta == TRUE) {
                D_inv_Sigma_beta <- mc_derivative_sigma_beta(
                    D = mu$D,
                    D_V_sqrt_mu = V_inv_sqrt$D_V_inv_sqrt_mu,
                    Omega = inv_Omega$inv_Omega,
                    V_sqrt = V_inv_sqrt$V_inv_sqrt, variance = variance)
                D_Sigma_beta <- lapply(D_inv_Sigma_beta,
                                       mc_sandwich_negative,
                                       bord1 = Sigma, bord2 = Sigma)
                output$D_Sigma_beta <- D_Sigma_beta
            }
        }
    }

    if (variance == "poisson_tweedie") {
        if (covariance == "identity" | covariance == "expm") {
            Omega <- mc_build_omega(tau = tau, Z = Z,
                                    covariance_link = covariance,
                                    sparse = sparse)
            V_sqrt <- mc_variance_function(
                mu = mu$mu, power = power,
                Ntrial = Ntrial, variance = "power", inverse = FALSE,
                derivative_power = !power_fixed,
                derivative_mu = compute_derivative_beta)
            Sigma <- forceSymmetric(Diagonal(length(mu$mu), mu$mu) +
                                        V_sqrt$V_sqrt %*%
                                        Omega$Omega %*% V_sqrt$V_sqrt)
            chol_Sigma <- chol(Sigma)
            inv_chol_Sigma <- solve(chol_Sigma)
            D_Sigma <- lapply(Omega$D_Omega, mc_sandwich,
                              bord1 = V_sqrt$V_sqrt,
                              bord2 = V_sqrt$V_sqrt)
            if (power_fixed == FALSE) {
                D_Sigma_power <- mc_sandwich_power(
                    middle = Omega$Omega,
                    bord1 = V_sqrt$V_sqrt, bord2 = V_sqrt$D_V_sqrt_p)
                D_Sigma <- c(D_Sigma_power = D_Sigma_power,
                             D_Sigma_tau = D_Sigma)
            }
            output <- list(Sigma_chol = chol_Sigma,
                           Sigma_chol_inv = inv_chol_Sigma,
                           D_Sigma = D_Sigma)
            if (compute_derivative_beta == TRUE) {
                D_Sigma_beta <- mc_derivative_sigma_beta(
                    D = mu$D,
                    D_V_sqrt_mu = V_sqrt$D_V_sqrt_mu, Omega$Omega,
                    V_sqrt = V_sqrt$V_sqrt, variance = variance)
                output$D_Sigma_beta <- D_Sigma_beta
            }
        }
        if (covariance == "inverse") {
            inv_Omega <- mc_build_omega(tau = tau, Z = Z,
                                        covariance_link = "inverse",
                                        sparse = sparse)
            Omega <- chol2inv(chol(inv_Omega$inv_Omega))
            V_sqrt <- mc_variance_function(
                mu = mu$mu, power = power,
                Ntrial = Ntrial, variance = "power", inverse = FALSE,
                derivative_power = !power_fixed,
                derivative_mu = compute_derivative_beta)
            D_Omega <- lapply(inv_Omega$D_inv_Omega,
                              mc_sandwich_negative, bord1 = Omega,
                              bord2 = Omega)
            D_Sigma <- lapply(D_Omega, mc_sandwich,
                              bord1 = V_sqrt$V_sqrt,
                              bord2 = V_sqrt$V_sqrt)
            Sigma <- forceSymmetric(Diagonal(length(mu$mu), mu$mu) +
                                        V_sqrt$V_sqrt %*% Omega %*%
                                        V_sqrt$V_sqrt)
            chol_Sigma <- chol(Sigma)
            inv_chol_Sigma <- solve(chol_Sigma)
            if (power_fixed == FALSE) {
                D_Sigma_p <- mc_sandwich_power(
                    middle = Omega,
                    bord1 = V_sqrt$V_sqrt,
                    bord2 = V_sqrt$D_V_sqrt_power)
                D_Sigma <- c(D_Sigma_p, D_Sigma)
            }
            output <- list(Sigma_chol = chol_Sigma,
                           Sigma_chol_inv = inv_chol_Sigma,
                           D_Sigma = D_Sigma)
            if (compute_derivative_beta == TRUE) {
                D_Sigma_beta <- mc_derivative_sigma_beta(
                    D = mu$D,
                    D_V_sqrt_mu = V_sqrt$D_V_sqrt_mu, Omega = Omega,
                    V_sqrt = V_sqrt$V_sqrt, variance = variance)
                output$D_Sigma_beta <- D_Sigma_beta
            }
        }
    }
    if(variance == "geom_tweedie") {
        if (covariance == "identity" | covariance == "expm") {
          Omega <- mc_build_omega(tau = tau, Z = Z,
                                  covariance_link = covariance,
                                  sparse = sparse)
          V_sqrt <- mc_variance_function(
            mu = mu$mu, power = power,
            Ntrial = Ntrial, variance = "power", inverse = FALSE,
            derivative_power = !power_fixed,
            derivative_mu = compute_derivative_beta)
          Sigma <- forceSymmetric(Diagonal(length(mu$mu), mu$mu^2) +
                                    V_sqrt$V_sqrt %*%
                                    Omega$Omega %*% V_sqrt$V_sqrt)
          chol_Sigma <- chol(Sigma)
          inv_chol_Sigma <- solve(chol_Sigma)
          D_Sigma <- lapply(Omega$D_Omega, mc_sandwich,
                            bord1 = V_sqrt$V_sqrt,
                            bord2 = V_sqrt$V_sqrt)
          if (power_fixed == FALSE) {
            D_Sigma_power <- mc_sandwich_power(
              middle = Omega$Omega,
              bord1 = V_sqrt$V_sqrt, bord2 = V_sqrt$D_V_sqrt_p)
            D_Sigma <- c(D_Sigma_power = D_Sigma_power,
                         D_Sigma_tau = D_Sigma)
          }
          output <- list(Sigma_chol = chol_Sigma,
                         Sigma_chol_inv = inv_chol_Sigma,
                         D_Sigma = D_Sigma)
          if (compute_derivative_beta == TRUE) {
            D_Sigma_beta <- mc_derivative_sigma_beta(
              D = mu$D,
              D_V_sqrt_mu = V_sqrt$D_V_sqrt_mu, Omega$Omega,
              V_sqrt = V_sqrt$V_sqrt, variance = variance)
            output$D_Sigma_beta <- D_Sigma_beta
          }
        }
        if (covariance == "inverse") {
          inv_Omega <- mc_build_omega(tau = tau, Z = Z,
                                      covariance_link = "inverse",
                                      sparse = sparse)
          Omega <- chol2inv(chol(inv_Omega$inv_Omega))
          V_sqrt <- mc_variance_function(
            mu = mu$mu, power = power,
            Ntrial = Ntrial, variance = "power", inverse = FALSE,
            derivative_power = !power_fixed,
            derivative_mu = compute_derivative_beta)
          D_Omega <- lapply(inv_Omega$D_inv_Omega,
                            mc_sandwich_negative, bord1 = Omega,
                            bord2 = Omega)
          D_Sigma <- lapply(D_Omega, mc_sandwich,
                            bord1 = V_sqrt$V_sqrt,
                            bord2 = V_sqrt$V_sqrt)
          Sigma <- forceSymmetric(Diagonal(length(mu$mu), mu$mu^2) +
                                    V_sqrt$V_sqrt %*% Omega %*%
                                    V_sqrt$V_sqrt)
          chol_Sigma <- chol(Sigma)
          inv_chol_Sigma <- solve(chol_Sigma)
          if (power_fixed == FALSE) {
            D_Sigma_p <- mc_sandwich_power(
              middle = Omega,
              bord1 = V_sqrt$V_sqrt,
              bord2 = V_sqrt$D_V_sqrt_power)
            D_Sigma <- c(D_Sigma_p, D_Sigma)
          }
          output <- list(Sigma_chol = chol_Sigma,
                         Sigma_chol_inv = inv_chol_Sigma,
                         D_Sigma = D_Sigma)
          if (compute_derivative_beta == TRUE) {
            D_Sigma_beta <- mc_derivative_sigma_beta(
              D = mu$D,
              D_V_sqrt_mu = V_sqrt$D_V_sqrt_mu, Omega = Omega,
              V_sqrt = V_sqrt$V_sqrt, variance = variance)
            output$D_Sigma_beta <- D_Sigma_beta
          }
        }
    }
    return(output)
}

mc_id <- function(data) {
  output <- list("Z0" = Diagonal(dim(data)[1],1))
  return(output)
}





mcglm <- function(linear_pred, matrix_pred, link, variance,
                  covariance, offset, Ntrial, power_fixed,
                  data, control_initial = "automatic",
                  contrasts = NULL, weights = NULL,
                  control_algorithm = list()) {
  n_resp <- length(linear_pred)
  linear_pred <- as.list(linear_pred)
  matrix_pred <- as.list(matrix_pred)
  if (missing(link)) {
    link <- rep("identity", n_resp)
  }
  if (missing(variance)) {
    variance <- rep("constant", n_resp)
  }
  if (missing(covariance)) {
    covariance <- rep("identity", n_resp)
  }
  if (missing(offset)) {
    offset <- rep(list(NULL), n_resp)
  }
  if (missing(Ntrial)) {
    Ntrial <- rep(list(rep(1, dim(data)[1])), n_resp)
  }
  if (missing(power_fixed)) {
    power_fixed <- rep(TRUE, n_resp)
  }
  if (missing(contrasts)) {
    contrasts <- NULL
  }
  link <- as.list(link)
  variance <- as.list(variance)
  covariance <- as.list(covariance)
  offset <- as.list(offset)
  Ntrial <- as.list(Ntrial)
  power_fixed <- as.list(power_fixed)
  if (class(control_initial) != "list") {
    control_initial <-
      mc_initial_values(linear_pred = linear_pred,
                        matrix_pred = matrix_pred, link = link,
                        variance = variance,
                        covariance = covariance, offset = offset,
                        Ntrial = Ntrial, contrasts = contrasts,
                        data = data)
    cat("Automatic initial values selected.", "\n")
  }
  con <- list(correct = TRUE, max_iter = 20, tol = 1e-04,
              method = "chaser", tuning = 1, verbose = FALSE)
  con[(namc <- names(control_algorithm))] <- control_algorithm
  list_model_frame <- lapply(linear_pred, model.frame, na.action = 'na.pass', data = data)
  if (!is.null(contrasts)) {
    list_X <- list()
    for (i in 1:n_resp) {
      options(na.action='na.pass')
      list_X[[i]] <- model.matrix(linear_pred[[i]],
                                  contrasts = contrasts[[i]])
      options(na.action='na.omit')
    }
  } else {
    options(na.action='na.pass')
    list_X <- lapply(linear_pred, model.matrix, data = data)
    options(na.action='na.omit')
  }
  list_Y <- lapply(list_model_frame, model.response)
  y_vec <- as.numeric(do.call(c, list_Y))
  if(is.null(weights)) {
    C <- rep(1, length(y_vec))
    C[is.na(y_vec)] = 0
    weights = C
    y_vec[is.na(y_vec)] <- 0
  }
  if(!is.null(weights)) {
    y_vec[is.na(y_vec)] <- 0
    if(class(weights) != "list") {weights <- as.list(weights)}
    weights <- as.numeric(do.call(c, weights))
  }
  sparse <- lapply(matrix_pred, function(x) {
    if (class(x) == "dgeMatrix") {
      FALSE
    } else TRUE
  })
  model_fit <- try(fit_mcglm(list_initial = control_initial,
                             list_link = link,
                             list_variance = variance,
                             list_covariance = covariance,
                             list_X = list_X, list_Z = matrix_pred,
                             list_offset = offset,
                             list_Ntrial = Ntrial,
                             list_power_fixed = power_fixed,
                             list_sparse = sparse, y_vec = y_vec,
                             correct = con$correct,
                             max_iter = con$max_iter, tol = con$tol,
                             method = con$method,
                             tuning = con$tuning,
                             verbose = con$verbose,
                             weights = weights))
  if (class(model_fit) != "try-error") {
    model_fit$beta_names <- lapply(list_X, colnames)
    model_fit$power_fixed <- power_fixed
    model_fit$list_initial <- control_initial
    model_fit$n_obs <- dim(data)[1]
    model_fit$link <- link
    model_fit$variance <- variance
    model_fit$covariance <- covariance
    model_fit$linear_pred <- linear_pred
    model_fit$con <- con
    model_fit$observed <- Matrix(y_vec, ncol = length(list_Y),
                                 nrow = dim(data)[1])
    model_fit$list_X <- list_X
    model_fit$matrix_pred <- matrix_pred
    model_fit$Ntrial <- Ntrial
    model_fit$offset <- offset
    model_fit$power_fixed
    model_fit$sparse <- sparse
    model_fit$data <- data
    model_fit$weights <- weights
    class(model_fit) <- "mcglm"
  }
  n_it <- length(na.exclude(model_fit$IterationCovariance[,1]))
  if(con$max_it == n_it) {warning("Maximum iterations number reached. \n", call. = FALSE)}
  return(model_fit)
}


mc_initial_values <- function(linear_pred, matrix_pred, link,
                              variance, covariance, offset,
                              Ntrial, contrasts = NULL, data) {
    n_resp <- length(linear_pred)
    if (!is.null(contrasts)) {
        list_X <- list()
        for (i in 1:n_resp) {
            list_X[[i]] <- model.matrix(linear_pred[[i]],
                                        contrasts = contrasts[[i]],
                                        data = data)
        }
    } else {
        list_X <- lapply(linear_pred, model.matrix, data = data)
    }
    list_models <- list()
    power_initial <- list()
    for (i in 1:n_resp) {
        if (variance[[i]] == "constant") {
            power_initial[[i]] <- 0
            if (!is.null(offset[[i]])) {
                data_temp <- data
                data_temp$offset <- offset[[i]]
                list_models[[i]] <-
                    glm(linear_pred[[i]],
                        family = quasi(link = link[[i]],
                                       variance =
                                           "constant"), offset = offset,
                        data = data_temp)
            } else {
                list_models[[i]] <-
                    glm(linear_pred[[i]],
                        family = quasi(link = link[[i]],
                                       variance = "constant"),
                        data = data)
            }
        }
        if (variance[[i]] == "tweedie" |
                variance[[i]] == "poisson_tweedie" |
            variance[[i]] == "geom_tweedie") {
            power_initial[[i]] <- 1
            if (!is.null(offset[[i]])) {
                data_temp <- data
                data_temp$offset <- offset[[i]]
                list_models[[i]] <-
                    glm(linear_pred[[i]],
                        family = quasi(link = link[[i]],
                                       variance = "mu"),
                        offset = offset, data = data_temp)
            } else {
                list_models[[i]] <-
                    glm(linear_pred[[i]],
                        family = quasi(link = link[[i]],
                                       variance = "mu"), data = data)
            }
        }
        if (variance[[i]] == "binomialP" |
                variance[[i]] == "binomialPQ") {
            power_initial[[i]] <- c(1)
            if (variance[[i]] == "binomialPQ") {
                power_initial[[i]] <- c(1, 1)
            }
            if (!is.null(Ntrial[[i]])) {
                temp <- model.frame(linear_pred[[i]], data = data)
                Y <- model.response(temp) * Ntrial[[i]]
                resp <- cbind(Y, Ntrial[[i]] - Y)
                X <- model.matrix(linear_pred[[i]], data = data)
                link_temp <- link[[i]]
                if (link_temp == "loglog") {
                    link_temp <- "cloglog"
                }
                list_models[[i]] <-
                    glm(resp ~ X - 1,
                        family = binomial(link = link_temp),
                        data = data)
            } else {
                link_temp <- link[[i]]
                if (link_temp == "loglog") {
                    link_temp <- "cloglog"
                }
                list_models[[i]] <-
                    glm(linear_pred[[i]],
                        family = quasi(link = link_temp,
                                       variance = "mu(1-mu)"),
                        data = data)
            }
        }
    }
    list_initial <- list()
    list_initial$regression <- lapply(list_models, coef)
    list_initial$power <- power_initial
    tau0_initial <- lapply(list_models,
                           function(x) summary(x)$dispersion)
    tau_extra <- lapply(matrix_pred, length)
    list_initial$tau <- list()
    for (i in 1:n_resp) {
        if (covariance[i] == "identity") {
            list_initial$tau[[i]] <-
                as.numeric(c(tau0_initial[[i]],
                             rep(0, c(tau_extra[[i]] - 1))))
        }
        if (covariance[i] == "inverse") {
            list_initial$tau[[i]] <-
                as.numeric(c(1/tau0_initial[[i]],
                             rep(0, c(tau_extra[[i]] - 1))))
        }
        if (covariance[i] == "expm") {
            list_initial$tau[[i]] <-
                as.numeric(c(log(tau0_initial[[i]]),
                             rep(0.1, c(tau_extra[[i]] - 1))))
        }
    }
    if (n_resp == 1) {
        list_initial$rho <- 0
    } else {
        list_initial$rho <- rep(0, n_resp * (n_resp - 1)/2)
    }
    return(list_initial)
}


mc_list2vec <- function(list_initial, list_power_fixed) {
    cov_ini <- do.call(c, Map(c, list_initial$power, list_initial$tau))
    n_resp <- length(list_initial$regression)
    indicadora <- list()
    for (i in 1:n_resp) {
        indicadora[[i]] <-
            rep(FALSE, length(list_initial$tau[[i]]))
    }
    indicadora_power <- list()
    for (i in 1:n_resp) {
        if (list_power_fixed[[i]] == FALSE) {
            indicadora_power[[i]] <-
                rep(FALSE, length(list_initial$power[[i]]))
        }
        if (list_power_fixed[[i]] == TRUE) {
            indicadora_power[[i]] <-
                rep(TRUE, length(list_initial$power[[i]]))
        }
    }
    index <- do.call(c, Map(c, indicadora_power, indicadora))
    cov_par <- data.frame(cov_ini, index)
    cov_ini <- cov_par[which(cov_par$index == FALSE), ]$cov_ini
    beta_ini <- do.call(c, list_initial$regression)
    cov_ini <- c(rho = list_initial$rho, cov_ini)
    return(list(beta_ini = as.numeric(beta_ini),
                cov_ini = as.numeric(cov_ini)))
}


mc_getInformation <- function(list_initial, list_power_fixed,
                              n_resp) {
    n_betas <- lapply(list_initial$regression, length)
    n_taus <- lapply(list_initial$tau, length)
    n_power <- lapply(list_initial$power, length)
    for (i in 1:n_resp) {
        if (list_power_fixed[[i]] == TRUE) {
            n_power[i] <- 0
        }
    }
    if (n_resp == 1) {
        n_rho <- 0
    }
    if (n_resp != 1) {
        n_rho <- length(list_initial$rho)
    }
    n_cov <- sum(do.call(c, n_power)) + n_rho +
        sum(do.call(c, n_taus))
    saida <- list(n_betas = n_betas, n_taus = n_taus, n_power = n_power,
                  n_rho = n_rho, n_cov = n_cov)
    return(saida)
}


mc_link_function <- function(beta, X, offset, link) {
    assert_that(noNA(beta))
    assert_that(noNA(X))
    if (!is.null(offset))
        assert_that(noNA(offset))
    link_name <- c("logit", "probit", "cauchit", "cloglog", "loglog",
                   "identity", "log", "sqrt", "1/mu^2", "inverse")
    link_func <- c("mc_logit", "mc_probit", "mc_cauchit", "mc_cloglog",
                   "mc_loglog", "mc_identity", "mc_log", "mc_sqrt",
                   "mc_invmu2", "mc_inverse")
    names(link_func) <- link_name
    if (!link %in% link_name) {
        ## Test if link function exists outside.
        if (!exists(link, envir = -1, mode = "function")) {
        stop(gettextf(paste0(
            "%s link function not recognised or found. ",
            "Available links are: ",
            paste(link_name, collapse = ", "),
            "."),
            sQuote(link)), domain = NA)
        } else {
            match_args <- sort(names(formals(link))) %in%
                sort(c("beta", "X", "offset"))
            ## Test if provided funtion has correct arguments.
            if (length(match_args) != 3L || !all(match_args)) {
                stop(gettextf(paste(
                    "Provided link function must have %s, %s and %s",
                    "as arguments to be valid."),
                    sQuote("beta"), sQuote("X"), sQuote("offset")),
                    domain = NA)
            }
        }
        output <- do.call(link,
                          args = list(beta = beta, X = X,
                                      offset = offset))
        if (!is.list(output)) {
            stop("Provided link funtion doesn't return a list.")
        }
        if (!identical(sort(names(output)), c("D","mu"))) {
            stop(paste0("Provided link funtion isn't return ",
                        "a list with names ", sQuote("mu"),
                        " and ", sQuote("D"), "."))
        }
        if (!(identical(dim(output$D), dim(X)) &&
              is.matrix(output$D))) {
            stop(paste0("Returned ", sQuote("D"),
                        " object by user defined link function ",
                        "isn't a matrix of correct dimensions."))
        }
        print(is.vector(output$mu, mode = "vector"))
        print(class(output$mu))
        if (!(length(output$mu) == nrow(X) &&
                  is.vector(output$mu, mode = "numeric"))) {
            stop(paste0("Returned ", sQuote("mu"),
                        " object by user defined link function ",
                        "isn't a vector of correct length."))
            is.vector(output$mu, mode = "vector")
        }
    } else {
        link <- link_func[link]
        output <- do.call(link,
                          args = list(beta = beta, X = X,
                                      offset = offset))
    }
    return(output)
}

#' @rdname mc_link_function
## Logit link function -------------------------------------------------
mc_logit <- function(beta, X, offset) {
    eta <- as.numeric(X %*% beta)
    if (!is.null(offset)) {
        eta <- eta + offset
    }
    mu <- make.link("logit")$linkinv(eta = eta)
    return(list(mu = mu, D = X * (mu * (1 - mu))))
}

#' @rdname mc_link_function
## Probit link function ------------------------------------------------
mc_probit <- function(beta, X, offset) {
    eta <- as.numeric(X %*% beta)
    if (!is.null(offset)) {
        eta <- eta + offset
    }
    mu <- make.link("probit")$linkinv(eta = eta)
    Deri <- make.link("probit")$mu.eta(eta = eta)
    return(list(mu = mu, D = X * Deri))
}

#' @rdname mc_link_function
## Cauchit link function -----------------------------------------------
mc_cauchit <- function(beta, X, offset) {
    eta <- as.numeric(X %*% beta)
    if (!is.null(offset)) {
        eta <- eta + offset
    }
    mu = make.link("cauchit")$linkinv(eta = eta)
    Deri <- make.link("cauchit")$mu.eta(eta = eta)
    return(list(mu = mu, D = X * Deri))
}

#' @rdname mc_link_function
## Complement log-log link function ------------------------------------
mc_cloglog <- function(beta, X, offset) {
    eta <- as.numeric(X %*% beta)
    if (!is.null(offset)) {
        eta <- eta + offset
    }
    mu = make.link("cloglog")$linkinv(eta = eta)
    Deri <- make.link("cloglog")$mu.eta(eta = eta)
    return(list(mu = mu, D = X * Deri))
}

#' @rdname mc_link_function
## Log-log link function -----------------------------------------------
mc_loglog <- function(beta, X, offset) {
    eta <- as.numeric(X %*% beta)
    if (!is.null(offset)) {
        eta <- eta + offset
    }
    mu <- exp(-exp(-eta))
    Deri <- exp(-exp(-eta) - eta)
    return(list(mu = mu, D = X * Deri))
}

#' @rdname mc_link_function
## Identity link function ----------------------------------------------
mc_identity <- function(beta, X, offset) {
    eta <- X %*% beta
    if (!is.null(offset)) {
        eta <- eta + offset
    }
    return(list(mu = as.numeric(eta), D = X))
}

#' @rdname mc_link_function
## Log link function ---------------------------------------------------
mc_log <- function(beta, X, offset) {
    eta <- as.numeric(X %*% beta)
    if (!is.null(offset)) {
        eta <- eta + offset
    }
    mu = make.link("log")$linkinv(eta = eta)
    return(list(mu = mu, D = X * mu))
}

#' @rdname mc_link_function
## Square-root link function -------------------------------------------
mc_sqrt <- function(beta, X, offset) {
    eta <- as.numeric(X %*% beta)
    if (!is.null(offset)) {
        eta <- eta + offset
    }
    mu = make.link("sqrt")$linkinv(eta = eta)
    return(list(mu = mu, D = X * (2 * as.numeric(eta))))
}

#' @rdname mc_link_function
## Inverse mu square link function -------------------------------------
mc_invmu2 <- function(beta, X, offset) {
    eta <- as.numeric(X %*% beta)
    if (!is.null(offset)) {
        eta <- eta + offset
    }
    mu <- make.link("1/mu^2")$linkinv(eta = eta)
    Deri <- make.link("1/mu^2")$mu.eta(eta = eta)
    return(list(mu = mu, D = X * Deri))
}

#' @rdname mc_link_function
## Inverse link function -----------------------------------------------
mc_inverse <- function(beta, X, offset) {
    eta <- as.numeric(X %*% beta)
    if (!is.null(offset)) {
        eta <- eta + offset
    }
    mu <- make.link("inverse")$linkinv(eta = eta)
    Deri <- make.link("inverse")$mu.eta(eta = eta)
    return(list(mu = mu, D = X * Deri))
}



mc_build_C <- function(list_mu, list_Ntrial, rho, list_tau, list_power,
                       list_Z, list_sparse, list_variance,
                       list_covariance, list_power_fixed,
                       compute_C = FALSE,
                       compute_derivative_beta = FALSE,
                       compute_derivative_cov = TRUE) {
    n_resp <- length(list_mu)
    n_obs <- length(list_mu[[1]][[1]])
    n_rho <- n_resp * (n_resp - 1)/2
    if (n_resp != 1) {
        assert_that(n_rho == length(rho))
    }
    list_Sigma_within <- suppressWarnings(
        Map(mc_build_sigma, mu = list_mu, Ntrial = list_Ntrial,
            tau = list_tau, power = list_power, Z = list_Z,
            sparse = list_sparse, variance = list_variance,
            covariance = list_covariance,
            power_fixed = list_power_fixed,
            compute_derivative_beta = compute_derivative_beta))
    list_Sigma_chol <- lapply(list_Sigma_within,
                              function(x) x$Sigma_chol)
    list_Sigma_inv_chol <- lapply(list_Sigma_within,
                                  function(x) x$Sigma_chol_inv)
    Sigma_between <- mc_build_sigma_between(rho = rho, n_resp = n_resp)
    II <- Diagonal(n_obs, 1)
    nucleo <- kronecker(Sigma_between$Sigmab, II)
    Bdiag_chol_Sigma_within <- bdiag(list_Sigma_chol)
    t_Bdiag_chol_Sigma_within <- t(Bdiag_chol_Sigma_within)
    Bdiag_inv_chol_Sigma <- bdiag(list_Sigma_inv_chol)
    inv_C <- Bdiag_inv_chol_Sigma %*%
        kronecker(solve(Sigma_between$Sigmab), II) %*%
        t(Bdiag_inv_chol_Sigma)
    output <- list(inv_C = inv_C)
    if (compute_derivative_cov == TRUE) {
        list_D_Sigma <- lapply(list_Sigma_within, function(x) x$D_Sigma)
        ## Derivatives of C with respect to power and tau parameters
        list_D_chol_Sigma <-
            Map(mc_derivative_cholesky, derivada = list_D_Sigma,
                inv_chol_Sigma = list_Sigma_inv_chol,
                chol_Sigma = list_Sigma_chol)
        mat_zero <- mc_build_bdiag(n_resp = n_resp, n_obs = n_obs)
        Bdiag_D_chol_Sigma <-
            mapply(mc_transform_list_bdiag,
                   list_mat = list_D_chol_Sigma,
                   response_number = 1:n_resp,
                   MoreArgs = list(mat_zero = mat_zero))
        Bdiag_D_chol_Sigma <- do.call(c, Bdiag_D_chol_Sigma)
        D_C <- lapply(Bdiag_D_chol_Sigma, mc_sandwich_cholesky,
                      middle = nucleo,
                      bord2 = t_Bdiag_chol_Sigma_within)
        ## Finish the derivatives with respect to power and tau
        ## parameters
        if (n_resp > 1) {
            D_C_rho <-
                mc_derivative_C_rho(D_Sigmab = Sigma_between$D_Sigmab,
                                    Bdiag_chol_Sigma_within =
                                        Bdiag_chol_Sigma_within,
                                    t_Bdiag_chol_Sigma_within =
                                        t_Bdiag_chol_Sigma_within,
                                    II = II)
            D_C <- c(D_C_rho, D_C)
        }
        output$D_C <- D_C
    }
    if (compute_C == TRUE) {
        C <- t_Bdiag_chol_Sigma_within %*%
            kronecker(Sigma_between$Sigmab, II) %*%
            Bdiag_chol_Sigma_within
        output$C <- C
    }
    if (compute_derivative_beta == TRUE) {
        list_D_Sigma_beta <- lapply(list_Sigma_within,
                                    function(x) x$D_Sigma_beta)
        list_D_chol_Sigma_beta <-
            Map(mc_derivative_cholesky, derivada = list_D_Sigma_beta,
                inv_chol_Sigma = list_Sigma_inv_chol,
                chol_Sigma = list_Sigma_chol)
        mat_zero <- mc_build_bdiag(n_resp = n_resp, n_obs = n_obs)
        Bdiag_D_chol_Sigma_beta <-
            mapply(mc_transform_list_bdiag,
                   list_mat = list_D_chol_Sigma_beta,
                   response_number = 1:n_resp,
                   MoreArgs = list(mat_zero = mat_zero))
        Bdiag_D_chol_Sigma_beta <- do.call(c, Bdiag_D_chol_Sigma_beta)
        D_C_beta <- lapply(Bdiag_D_chol_Sigma_beta,
                           mc_sandwich_cholesky, middle = nucleo,
                           bord2 = t_Bdiag_chol_Sigma_within)
        output$D_C_beta <- D_C_beta
    }
    return(output)
}


mc_build_omega <- function(tau, Z, covariance_link, sparse = FALSE) {
    if (covariance_link == "identity") {
        Omega <- mc_matrix_linear_predictor(tau = tau, Z = Z)
        output <- list(Omega = Omega, D_Omega = Z)
    }
    if (covariance_link == "expm") {
        U <- mc_matrix_linear_predictor(tau = tau, Z = Z)
        temp <- mc_expm(U = U, inverse = FALSE, sparse = sparse)
        D_Omega <- lapply(Z, mc_derivative_expm, UU = temp$UU,
                          inv_UU = temp$inv_UU, Q = temp$Q, sparse = sparse)
        output <- list(Omega = forceSymmetric(temp$Omega),
                       D_Omega = D_Omega)
    }
    if (covariance_link == "inverse") {
        inv_Omega <- mc_matrix_linear_predictor(tau = tau, Z = Z)
        output <- list(inv_Omega = inv_Omega, D_inv_Omega = Z)
    }
    return(output)
}



mc_matrix_linear_predictor <- function(tau, Z) {
    if (length(Z) != length(tau)) {
        stop("Incorrect number of parameters")
    }
    output <- mapply("*", Z, tau, SIMPLIFY = FALSE)
    output <- Reduce("+", output)
    return(output)
}


mc_variance_function <- function(mu, power, Ntrial,
                                 variance, inverse,
                                 derivative_power,
                                 derivative_mu) {
    assert_that(is.logical(inverse))
    assert_that(is.logical(derivative_power))
    assert_that(is.logical(derivative_mu))
    switch(variance,
           power = {
               output <- mc_power(mu = mu, power = power,
                                  inverse = inverse,
                                  derivative_power = derivative_power,
                                  derivative_mu = derivative_mu)
           },
           binomialP = {
               output <- mc_binomialP(mu = mu, power = power,
                                      Ntrial = Ntrial,
                                      inverse = inverse,
                                      derivative_power =
                                          derivative_power,
                                      derivative_mu = derivative_mu)
           },
           binomialPQ = {
               output <- mc_binomialPQ(mu = mu, power = power,
                                       Ntrial = Ntrial,
                                       inverse = inverse,
                                       derivative_power =
                                           derivative_power,
                                       derivative_mu = derivative_mu)
           },
           stop(gettextf("%s variance function not recognised",
                         sQuote(variance)), domain = NA))
    return(output)
}

#' @rdname mc_variance_function
## Power variance function ---------------------------------------------
mc_power <- function(mu, power, inverse,
                     derivative_power,
                     derivative_mu) {
    ## The observed value can be zero, but not the expected value.
    assert_that(all(mu > 0))
    assert_that(is.number(power))
    mu.power <- mu^power
    sqrt.mu.power <- sqrt(mu.power)
    n <- length(mu)
    if (inverse == TRUE & derivative_power == TRUE &
            derivative_mu == FALSE) {
        output <- list(
            V_inv_sqrt = Diagonal(n = n, 1/sqrt.mu.power),
            D_V_inv_sqrt_power =
                Diagonal(n = n,
                         -(mu.power * log(mu))/(2 * (mu.power)^(1.5))))
    }
    if (inverse == TRUE & derivative_power == FALSE &
            derivative_mu == FALSE) {
        output <- list(V_inv_sqrt = Diagonal(n = n, 1/sqrt.mu.power))
    }
    if (inverse == FALSE & derivative_power == TRUE &
            derivative_mu == FALSE) {
        output <- list(
            V_sqrt = Diagonal(n = n, sqrt.mu.power),
            D_V_sqrt_power =
                Diagonal(n = n,
                         +(mu.power * log(mu))/(2 * sqrt.mu.power)))
    }
    if (inverse == FALSE & derivative_power == FALSE &
            derivative_mu == FALSE) {
        output <- list(V_sqrt = Diagonal(n = n, sqrt.mu.power))
    }
    if (inverse == TRUE & derivative_power == TRUE &
            derivative_mu == TRUE) {
        output <- list(
            V_inv_sqrt = Diagonal(n = n, 1/sqrt.mu.power),
            D_V_inv_sqrt_power =
                Diagonal(n = n,
                         -(mu.power * log(mu))/(2 * (mu.power)^(1.5))),
            D_V_inv_sqrt_mu = -(mu^(power -  1) * power)/
                                   (2 * (mu.power)^(1.5)))
    }
    if (inverse == TRUE & derivative_power == FALSE &
            derivative_mu == TRUE) {
        output <- list(
            V_inv_sqrt = Diagonal(n = n, 1/sqrt.mu.power),
            D_V_inv_sqrt_mu = -(mu^(power - 1) * power)/
                                   (2 * (mu.power)^(1.5)))
    }
    if (inverse == FALSE & derivative_power == TRUE &
            derivative_mu == TRUE) {
        output <- list(
            V_sqrt = Diagonal(n = n, sqrt.mu.power),
            D_V_sqrt_power =
                Diagonal(n = n, (mu.power * log(mu))/
                                    (2 * sqrt.mu.power)),
            D_V_sqrt_mu = (mu^(power - 1) * power)/(2 * sqrt.mu.power))
    }
    if (inverse == FALSE & derivative_power == FALSE &
            derivative_mu == TRUE) {
        output <- list(V_sqrt = Diagonal(n = n, sqrt.mu.power),
                       D_V_sqrt_mu = (mu^(power - 1) * power)/
                                         (2 * sqrt.mu.power))
    }
    return(output)
}

#' @rdname mc_variance_function
#' @usage mc_binomialP(mu, power, inverse, Ntrial,
#'                     derivative_power, derivative_mu)
## BinomialP variance function
## -----------------------------------------
mc_binomialP <- function(mu, power, inverse, Ntrial,
                         derivative_power,
                         derivative_mu) {
    ## The observed value can be 0 and 1, but not the expected value
    assert_that(all(mu > 0))
    assert_that(all(mu < 1))
    assert_that(is.number(power))
    assert_that(all(Ntrial > 0))
    constant <- (1/Ntrial)
    mu.power <- mu^power
    mu.power1 <- (1 - mu)^power
    mu1mu <- constant * (mu.power * mu.power1)
    sqrt.mu1mu <- sqrt(mu1mu)
    n <- length(mu)
    if (inverse == TRUE & derivative_power == TRUE &
            derivative_mu == FALSE) {
        output <- list(
            V_inv_sqrt = Diagonal(n = n, 1/sqrt.mu1mu),
            D_V_inv_sqrt_power =
                Diagonal(n = n, -(log(1 - mu) * mu1mu +
                                  log(mu) * mu1mu)/(2 * (mu1mu^(1.5)))))
    }
    if (inverse == TRUE & derivative_power == FALSE &
            derivative_mu == FALSE) {
        output <- list(V_inv_sqrt = Diagonal(n = n, 1/sqrt.mu1mu))
    }
    if (inverse == FALSE & derivative_power == TRUE &
            derivative_mu == FALSE) {
        output <- list(
            V_sqrt = Diagonal(n = n, sqrt.mu1mu),
            D_V_sqrt_power = Diagonal(n = n, (log(1 - mu) * mu1mu +
                                              log(mu) * mu1mu)/
                                                 (2 * sqrt.mu1mu)))
    }
    if (inverse == FALSE & derivative_power == FALSE &
            derivative_mu == FALSE) {
        output <- list(V_sqrt = Diagonal(n = n, sqrt.mu1mu))
    }
    if (inverse == TRUE & derivative_power == TRUE &
            derivative_mu == TRUE) {
        output <- list(
            V_inv_sqrt = Diagonal(n = n, 1/sqrt.mu1mu),
            D_V_inv_sqrt_power =
                Diagonal(n = n, -(log(1 - mu) * mu1mu + log(mu) *
                                  mu1mu)/(2 * (mu1mu^(1.5)))),
            D_V_inv_sqrt_mu = -(constant * (mu.power1 *
                                            (mu^(power - 1)) * power) -
                                constant * (((1 - mu)^(power - 1)) *
                                            mu.power * power))/
                                   (2 * (mu1mu^(1.5))))
    }
    if (inverse == TRUE & derivative_power == FALSE &
            derivative_mu == TRUE) {
        output <- list(
            V_inv_sqrt = Diagonal(n = n, 1/sqrt.mu1mu),
            D_V_inv_sqrt_mu = -(constant *
                                (mu.power1 * (mu^(power - 1)) * power) -
                                constant * (((1 - mu)^(power - 1)) *
                                            mu.power * power))/
                                   (2 * (mu1mu^(1.5))))
    }
    if (inverse == FALSE & derivative_power == TRUE &
            derivative_mu == TRUE) {
        output <- list(
            V_sqrt = Diagonal(n = n, sqrt.mu1mu),
            D_V_sqrt_power = Diagonal(n = n, (log(1 - mu) * mu1mu +
                                              log(mu) * mu1mu)/
                                                 (2 * sqrt.mu1mu)),
            D_V_sqrt_mu = (constant *
                           (mu.power1 * (mu^(power - 1)) * power) -
                           constant * (((1 - mu)^(power - 1)) *
                                       mu.power * power))/
                              (2 * sqrt.mu1mu))
    }
    if (inverse == FALSE & derivative_power == FALSE &
            derivative_mu == TRUE) {
        output <- list(
            V_sqrt = Diagonal(n = n, sqrt.mu1mu),
            D_V_sqrt_mu = (constant *
                           (mu.power1 * (mu^(power - 1)) * power) -
                           constant * (((1 - mu)^(power - 1)) *
                                       mu.power * power))/
                              (2 * sqrt.mu1mu))
    }
    return(output)
}

#' @rdname mc_variance_function
#' @usage mc_binomialPQ(mu, power, inverse, Ntrial,
#'                      derivative_power, derivative_mu)
## BinomialPQ variance function ----------------------------------------
mc_binomialPQ <- function(mu, power, inverse,
                          Ntrial, derivative_power,
                          derivative_mu) {
    ## The observed value can be 0 and 1, but not the expected value
    assert_that(all(mu > 0))
    assert_that(all(mu < 1))
    assert_that(length(power) == 2)
    assert_that(all(Ntrial > 0))
    constant <- (1/Ntrial)
    p <- power[1]
    q <- power[2]
    mu.p <- mu^p
    mu1.q <- (1 - mu)^q
    mu.p.mu.q <- mu.p * mu1.q
    mu1mu <- mu.p.mu.q * constant
    sqrt.mu1mu <- sqrt(mu1mu)
    n <- length(mu)
    if (inverse == TRUE & derivative_power == TRUE &
            derivative_mu == FALSE) {
        denominator <- (2 * (mu1mu^1.5) * Ntrial)
        output <- list(
            V_inv_sqrt = Diagonal(n = n, 1/sqrt.mu1mu),
            D_V_inv_sqrt_p = Diagonal(n = n,
                                      -(mu.p.mu.q * log(mu))/
                                           denominator),
            D_V_inv_sqrt_q = Diagonal(n = n,
                                      -mu.p.mu.q * log(1 - mu)/
                                           denominator))
    }
    if (inverse == TRUE & derivative_power == FALSE &
            derivative_mu == FALSE) {
        output <- list(V_inv_sqrt = Diagonal(n = n, 1/sqrt.mu1mu))
    }
    if (inverse == FALSE & derivative_power == TRUE &
            derivative_mu == FALSE) {
        denominator <- 2 * sqrt.mu1mu * Ntrial
        output <- list(
            V_sqrt = Diagonal(n = n, sqrt.mu1mu),
            D_V_sqrt_p = Diagonal(n = n,
                                  +(mu.p.mu.q * log(mu))/denominator),
            D_V_sqrt_q = Diagonal(n = n,
                                  +(mu.p.mu.q * log(1 - mu))/
                                       denominator))
    }
    if (inverse == FALSE & derivative_power == FALSE &
            derivative_mu == FALSE) {
        output <- list(V_sqrt = Diagonal(n = n, sqrt.mu1mu))
    }
    if (inverse == TRUE & derivative_power == TRUE &
            derivative_mu == TRUE) {
        denominator <- (2 * (mu1mu^1.5) * Ntrial)
        output <- list(
            V_inv_sqrt = Diagonal(n = n, 1/sqrt.mu1mu),
            D_V_inv_sqrt_p = Diagonal(n = n,
                                      -(mu.p.mu.q * log(mu))/
                                           denominator),
            D_V_inv_sqrt_q = Diagonal(n = n,
                                      -mu.p.mu.q *
                                           log(1 - mu)/denominator),
            D_V_inv_sqrt_mu = -(constant *
                                (mu1.q * (mu^(p - 1)) * p) -
                                constant * (((1 - mu)^(q - 1)) *
                                            mu.p * q))/
                                   (2 * (mu1mu^1.5)))
    }
    if (inverse == TRUE & derivative_power == FALSE &
            derivative_mu == TRUE) {
        output <- list(
            V_inv_sqrt = Diagonal(n = n, 1/sqrt.mu1mu),
            D_V_inv_sqrt_mu = -(constant * (mu1.q * (mu^(p - 1)) * p) -
                                constant * (((1 - mu)^(q - 1)) *
                                            mu.p * q))/
                                   (2 * (mu1mu^1.5)))
    }
    if (inverse == FALSE & derivative_power == TRUE &
            derivative_mu == TRUE) {
        denominator1 <- 2 * sqrt.mu1mu
        denominator2 <- denominator1 * Ntrial
        output <- list(
            V_sqrt = Diagonal(n = n, sqrt.mu1mu),
            D_V_sqrt_p = Diagonal(n = n, (mu.p.mu.q * log(mu))/
                                             denominator2),
            D_V_sqrt_q = Diagonal(n = n, (mu.p.mu.q * log(1 - mu))/
                                             denominator2),
            D_V_sqrt_mu = (constant * (mu1.q * (mu^(p - 1)) * p) -
                           constant * (((1 - mu)^(q - 1)) * mu.p * q))/
                              denominator1)
    }
    if (inverse == FALSE & derivative_power == FALSE &
            derivative_mu == TRUE) {
        output <- list(
            V_sqrt = Diagonal(n = n, sqrt.mu1mu),
            D_V_sqrt_mu = (constant * (mu1.q * (mu^(p - 1)) * p) -
                           constant * (((1 - mu)^(q - 1)) * mu.p * q))/
                              (2 * sqrt.mu1mu))
    }
    return(output)
}



mc_sandwich <- function(middle, bord1, bord2) {
    bord1 %*% middle %*% bord2
}

#' @rdname mc_sandwich
mc_sandwich_negative <- function(middle, bord1, bord2) {
    -bord1 %*% middle %*% bord2
}

#' @rdname mc_sandwich
mc_sandwich_power <- function(middle, bord1, bord2) {
    temp1 <- mc_sandwich(middle = middle, bord1 = bord1, bord2 = bord2)
    return(temp1 + t(temp1))
}

#' @rdname mc_sandwich
mc_sandwich_cholesky <- function(bord1, middle, bord2) {
    p1 <- bord1 %*% middle %*% bord2
    return(p1 + t(p1))
}

#' @rdname mc_sandwich
mc_multiply <- function(bord1, bord2) {
    return(bord2 %*% bord1)
}

#' @rdname mc_sandwich
mc_multiply2 <- function(bord1, bord2) {
    return(bord1 %*% bord2)
}


mc_build_sigma_between <- function(rho, n_resp, inverse = FALSE) {
    output <- list(Sigmab = 1, D_Sigmab = 1)
    if (n_resp > 1) {
        Sigmab <- Diagonal(n_resp, 1)
        Sigmab[lower.tri(Sigmab)] <- rho
        Sigmab <- forceSymmetric(t(Sigmab))
        D_Sigmab <- mc_derivative_sigma_between(n_resp = n_resp)
        if (inverse == FALSE) {
            output <- list(Sigmab = Sigmab, D_Sigmab = D_Sigmab)
        }
        if (inverse == TRUE) {
            inv_Sigmab <- solve(Sigmab)
            D_inv_Sigmab <- lapply(D_Sigmab, mc_sandwich_negative,
                                   bord1 = inv_Sigmab,
                                   bord2 = inv_Sigmab)
            output <- list(inv_Sigmab = inv_Sigmab,
                           D_inv_Sigmab = D_inv_Sigmab)
        }
    }
    return(output)
}

#' @rdname mc_build_sigma_between
mc_derivative_sigma_between <- function(n_resp) {
    position <- combn(n_resp, 2)
    list.Derivative <- list()
    n_par <- n_resp * (n_resp - 1)/2
    for (i in 1:n_par) {
        Derivative <- Matrix(0, ncol = n_resp, nrow = n_resp)
        Derivative[position[1, i], position[2, i]] <-
            Derivative[position[2, i], position[1, i]] <- 1
        list.Derivative[i][[1]] <- Derivative
    }
    return(list.Derivative)
}


mc_quasi_score <- function(D, inv_C, y_vec, mu_vec, W) {
    res <- y_vec - mu_vec
    t_D <- t(D)
    part1 <- t_D %*% inv_C
    score <- part1 %*% W %*% res
    sensitivity <- -part1 %*% W %*% D
    variability <- part1 %*% W^2 %*% D
    output <- list(Score = score, Sensitivity = sensitivity,
                   Variability = variability)
    return(output)
}


mc_updateBeta <- function(list_initial, betas, information, n_resp) {
    cod <- rep(1:n_resp, information$n_betas)
    temp <- data.frame(beta = betas, cod)
    for (k in 1:n_resp) {
        list_initial$regression[[k]] <-
            temp[which(temp$cod == k), ]$beta
    }
    return(list_initial)
}


mc_derivative_cholesky <- function(derivada, inv_chol_Sigma,
                                   chol_Sigma) {
    faux <- function(derivada, inv_chol_Sigma, chol_Sigma) {
        t1 <- inv_chol_Sigma %*% derivada %*% t(inv_chol_Sigma)
        t1 <- tril(t1)
        diag(t1) <- diag(t1)/2
        output <- chol_Sigma %*% t1
        return(output)
    }
    list_D_chol <- lapply(derivada, faux,
                          inv_chol_Sigma = inv_chol_Sigma,
                          chol_Sigma = chol_Sigma)
    return(list_D_chol)
}


mc_build_bdiag <- function(n_resp, n_obs) {
    list_zero <- list()
    for (i in 1:n_resp) {
        list_zero[[i]] <- Matrix(0, n_obs, n_obs, sparse = TRUE)
    }
    return(list_zero)
}


mc_transform_list_bdiag <- function(list_mat, mat_zero,
                                    response_number) {
    aux.f <- function(x, mat_zero, response_number) {
        mat_zero[[response_number]] <- x
        return(bdiag(mat_zero))
    }
    output <- lapply(list_mat, aux.f, mat_zero = mat_zero,
                     response_number = response_number)
    return(output)
}


mc_pearson <- function(y_vec, mu_vec, Cfeatures, inv_J_beta = NULL,
                       D = NULL, correct = FALSE,
                       compute_sensitivity = TRUE,
                       compute_variability = FALSE,
                       W) {
    product <- lapply(Cfeatures$D_C, mc_multiply,
                      bord2 = Cfeatures$inv_C)
    res <- y_vec - mu_vec
    pearson_score <- unlist(lapply(product, mc_core_pearson,
                                   inv_C = Cfeatures$inv_C, res = res, W = W))

    sensitivity <- matrix(NA, length(product), length(product))
    if(compute_sensitivity == TRUE) {
      sensitivity <- mc_sensitivity(product, W = W)
    }

    output <- list(Score = pearson_score, Sensitivity = sensitivity,
                   Extra = product)
    if (correct == TRUE) {
        correction <- mc_correction(D_C = Cfeatures$D_C,
                                    inv_J_beta = inv_J_beta, D = D,
                                    inv_C = Cfeatures$inv_C)
        output <- list(Score = pearson_score + correction,
                       Sensitivity = sensitivity, Extra = product)
    }
    if (compute_variability == TRUE) {
        variability <- mc_variability(sensitivity = sensitivity,
                                      product = product,
                                      inv_C = Cfeatures$inv_C,
                                      C = Cfeatures$C, res = res, W = W)
        output$Variability <- variability
    }
    return(output)
}


mc_core_pearson <- function(product, inv_C, res, W) {
    product <- product %*% W
    output <- t(res) %*% product %*%
        (inv_C %*% res) - sum(diag(product))
    return(as.numeric(output))
}


mc_sensitivity <- function(product, W) {
    #sourceCpp("src/mc_sensitivity_op.cpp")
    Sensitivity <- mc_sensitivity_op(products = product, W = W)
    Sensitivity <- forceSymmetric(Sensitivity, uplo = FALSE)
    return(Sensitivity)
}

mc_sensitivity_op <- function(products, W) {
    .Call('_mcglm_mc_sensitivity_op', PACKAGE = 'mcglm', products, W)
}

mc_variability_op <- function(sensitivity, WE, k4, W) {
    .Call('_mcglm_mc_variability_op', PACKAGE = 'mcglm', sensitivity, WE, k4, W)
}

mc_correction <- function(D_C, inv_J_beta, D, inv_C) {
    term1 <- lapply(D_C, mc_sandwich, bord1 = t(D) %*% inv_C,
                    bord2 = inv_C %*% D)
    output <- lapply(term1,
                     function(x, inv_J_beta) sum(x * inv_J_beta),
                     inv_J_beta = inv_J_beta)
    return(-unlist(output))
}

mc_updateCov <- function(list_initial, covariance, list_power_fixed,
                         information, n_resp) {
    rho_cod <- rep("rho", information$n_rho)
    tau_cod <- list()
    power_cod <- list()
    for (i in 1:n_resp) {
        power_cod[[i]] <- rep(paste("power", i, sep = ""),
                              information$n_power[[i]])
        tau_cod[[i]] <- rep(paste("tau", i, sep = ""),
                            information$n_tau[[i]])
    }
    temp <- data.frame(values = covariance,
                       cod = c(rho_cod,
                               do.call(c, Map(c, power_cod, tau_cod))))
    cod.tau <- paste("tau", 1:n_resp, sep = "")
    for (i in 1:n_resp) {
        list_initial$tau[[i]] <-
            temp[which(temp$cod == cod.tau[i]), ]$values
    }
    cod.power <- paste("power", 1:n_resp, sep = "")
    for (i in 1:n_resp) {
        if (list_power_fixed[[i]] == FALSE) {
            list_initial$power[[i]] <-
                temp[which(temp$cod == cod.power[i]), ]$values
        }
    }
    if (length(information$n_betas) != 1) {
        list_initial$rho <-
            temp[which(temp$cod == "rho"), ]$values
    }
    return(list_initial)
}