---
title: 'Multivariate Covariance Generalized Linear Models in Python: The mcglm library'
tags:
  - Python
  - statistical models
  - multivariate statistical analysis
  - longitudinal data analysis
  - MCGLM
  - GLM
  - statsmodels
authors:
  - name: Jean Carlos Faoot Maia
    equal-contrib: true
    orcid: 0009-0001-3747-0669
    affiliation: 1
  - name: Wagner Hugo Bonat
    orcid: 0000-0002-0349-7054
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Paraná Federal University, Brazil
   index: 1
date: 9 August 2023
bibliography: paper.bib

---

# Abstract

The `mcglm` library, a newly introduced Python tool, facilitates statistical 
analyses using Multivariate Covariance Generalized Linear Models (McGLM). This 
contemporary family of models universalizes the traditional Generalized Linear 
Models (GLM), empowering them to handle multivariate and non-independent response 
variables. Due to its flexibility and explicit specification, McGLM supports a 
lot of statistical analyses across different kinds of data and distinct traits; in 
this article, we promote `mcglm`.

The `mcglm` library provides a comprehensive platform for data analysis using 
the McGLM framework. Built upon the established standards of the `statsmodels` 
library, it provides a comprehensive summary report for fitting assessment, 
including elementary parameters such as regression and dispersion coefficients, 
confidence intervals, hypothesis testing results, residual analysis, 
goodness-of-fit measurements, and correlation coefficients between outcomes. 
In addition, the library provides a rich set of link and variance functions 
and tools to define inner-response dependency matrices through the matrix 
linear predictor. The base code is extendable and reliable, reflecting sound 
object-oriented programming practices and thorough unit testing. 

The library is hosted on PyPI and can be installed with some Python library 
manager, such as `pip`.

# Introduction

Dated at the beginning of the 19th century and controversial about the actual 
authorship, the least squares method established an optimization algorithm 
[@10.1214/aos/1176345451]. According to the Gauss-Markov theorem [@Gauss-Marc], 
the resulting estimates are optimal and unbiased under linear conditions. 
This optimization method forms the basis of linear regression, one of the 
earliest statistical models [@galton:1886; @linearregression:1982]. Linear 
regression associates a response variable to a group of covariates by 
employing a linear operation on regression coefficients [@10.2307/2333849]. 
Three main assumptions for linear regression are linearity, independent 
realizations of the response variable, and a Gaussian homoscedastic error 
with a zero mean. While standing the test of time, linear regression has 
motivated numerous statistical proposals seeking to generalize its 
foundational assumptions.

Over time, many statistical proposals have aimed to extend the linear 
regression. @glm:1972 proposed the Generalized Linear Model (GLM), which 
relieves the Gaussian assumption accommodating exponential family models 
[@GLM:2004]. Similarly, the Generalized Additive Model (GAM) [@GAM:1986] 
eases the linear assumption by using covariates smooth functions. The 
Generalized Estimating Equations (GEE) [@Zeger:1988] apply the 
quasi-likelihood estimating functions to deal with dependent data. Additional 
consolidated frameworks for dependent data include Copulas [@Krupskii:2013; 
@Masarotto:2012], and Mixed Models [@Verbeke:2014], among others. One 
prevalent aspect of the cited frameworks is that they cannot deal with 
multiple response variables.

The Multivariate Covariance Generalized Linear Model (McGLM) [@Bonat:2016] 
extends the GLM by allowing the multivariate analysis of non-independent 
responses, including longitudinal and spatial data. The model is defined 
through second-moment assumptions, utilizing five essential components: the 
linear predictor via design matrix, link function, variance function, 
covariance link function, and the matrix linear predictor. As a comprehensive 
statistical model, McGLM facilitates analysis by assessing regression and 
dispersion coefficients, hypothesis tests, goodness-of-fit measurements, 
and correlation coefficients between response variables. To the best of our 
knowledge, the library `mcglm` is the first holistic framework to support 
statistical analysis in Python with the aid of McGLM.

# Statement of need

The McGLM framework is available for R users through the open-source package 
`mcglm` [@Bonat:2016b]; nevertheless, the language Python did not have a 
standard library until the library `mcglm`. The foremost library statistical 
analysis in Python is the `statsmodels` [@Seabold:2010]. It implements 
classical statistical models, such as GLM, GAM, GEE, and Copulas. Many other 
libraries stand out for probabilistic programming in Python [@probabilisticp:2018], 
such as: `PyMC` [@pymc3:2016], `Pyro` [@pyro:2018], and `PyStan` [@stan:2017]. 
Those libraries distinguish from `statsmodels` on their Bayesian paradigm 
of specifying models. The library `mcglm` specifies the McGLM in a frequentist 
fashion.

The library `mcglm` provides an easy interface for fitting McGLM on the 
standards of the `statsmodels` [@Seabold:2010] library. It provides a 
comprehensive framework for statistical analysis supported by McGLM, 
with tools to lead its model specification, fitting, and appropriate report 
to assess estimates.

# Model Components

McGLM is specified by five components: linear predictors, link 
functions, variance functions, matrix linear predictors, and covariance 
link functions. In this section, we discuss the usual choice for each 
of these components.

McGLM offers the flexibility to specify typical linear predictors, 
including the usual formula notation popular in many statistical 
software. In alignment with the GLM framework, the link function 
encompasses usual choices like logit and probit for binary and 
binomial data, log for count data, and identity for continuous 
accurate data. The variance function is fundamental to the McGLM, 
as it is related to the marginal distribution of the response 
variable. Noteworthy among common choices is the power of the 
variance function, which is specialized for handling continuous 
data and defines the Tweedie family of distributions, as elucidated 
by @bent:1987 and @bent:1997. This family includes exceptional cases 
such as Gaussian (p = 0), Gamma (p = 2), and Inverse Gaussian (p = 3). 
The variance function extended binomial is a common choice for 
analyzing bounded data. For fitting count data, the dispersion 
function presented by @kokonendji:2015, called Poisson-Tweedie, is 
flexible enough to capture notable models, such as Hermite (p = 0), 
Neyman Type A (p = 1), Negative Binomial (p = 2) and Poisson inverse 
Gaussian (p = 3). The following table summarizes the mentioned variance 
functions:

\begin{table}[h]
\centering
\label{tab:methods}
\begin{tabular}{ll} \hline
Function name            & Formula  \\ \hline
\texttt{Tweedie/Power}  & $\mathrm{V}(.;p) = \mu^{p}$\\
\texttt{Binomial}   & $\mathrm{V}(.;p) = \mu^{p} (1 - \mu)^{p}$\\
\texttt{Poisson-Tweedie}   & $\mathrm{V}(.;p) = \mu + \mu^{p}$\\ \hline
\end{tabular}
\caption{Table with variance functions implemented}
\end{table}

The user specifies the dependency through the Z matrices in the matrix 
linear predictor to describe the covariance structure. Many of the 
classical statistical models are replicable by setting tailored Z 
matrices. To cite a few, mixed models, moving averages, and compound 
symmetry. For more details, see @Bonat:2016 and @Bonat:2018. Finally, 
@Bonat:2018 proposed three covariance link functions: identity, 
inverse, and exponential-matrix.

# The Python library mcglm

![UML for the library \label{fig:umlcode}](artifacts/classes.png)

The library `mcglm` provides the first Python tool for statistical 
analysis with the aid of McGLM. Heavily influenced by its twin R 
version [@Bonat:2018], the library has ninety-one percent of 
unit-testing coverage. URLs of source-code and PyPI, the official 
repository for Python libraries, are [https://github.com/jeancmaia/mcglm] 
(https://github.com/jeancmaia/mcglm) and [https://pypi.org/project/mcglm/]
(https://pypi.org/project/mcglm/). The library `mcglm` can easily be installed 
using the library `pip.`

The `mcglm` library is based on popular libraries of scientific Python 
programming: The `NumPy` [@harris2020array], `scipy` [@2020SciPy-NMeth], 
and `scipy.sparse`. We inherit `statsmodels`'s interface and deliver a 
code library akin to their standards API. Object-oriented programming 
is another cornerstone for the library `mcglm`; the SOLID principles 
[@Madasu:2015] helped to create a readable and extensible code base. The 
UML diagram \autoref{fig:umlcode} presents the `mcglm` library architecture.

The implementation `mcglm` lies in six classes: `MCGLM`, `MCGLMMean`, 
`MCGLMVariance`, `MCGLMCAttributes`, `MCGLMParameters` and `MCGLMResults`. 
Each class has its scope and responsibilities. For in-depth details, access 
the code-base [https://github.com/jeancmaia/mcglm](https://github.com/jeancmaia/mcglm).

We adopted the `statsmodels` standards of attribute names; the endog argument 
is a vector, or a matrix, with the realizations of the response variable; the 
exog statement defines the covariates through design matrices. For multiple 
outcomes, endog and exog must be specified via Python lists. The z argument 
establishes dependency arrays through `numpy` array structures.

Arguments link and variance set the link and variance functions, respectively. 
For the former, the available options are Identity, Logit, Power, Log, Probit, 
Cauchy, Cloglog, Log-log, NegativeBinomial, and Inverse Power - all canonical 
options for GLM. Suitable options for the variance are Constant, Tweedie, 
BinomialP, BinomialPQ, Geometric-Tweedie, and Poisson-Tweedie. The default 
values for the link and variance functions are identity and constant, suitable 
picks for Gaussian models. For multiple outcomes, link and variance must be 
specified via Python lists.

The offset argument is suitable for either continuous or count outcomes. In 
addition, parameter ntrial is the canonical number of trials for binomial 
data. Finally, parameter power_fixed activates searching for the power 
parameter for Tweedie models. For multiple outcomes, parameters must be 
specified via Python lists.

An instantiated object can fit a model with the fit() method, which 
returns an object of the `MCGLMResults` class. This object can trigger two 
methods: summary(), a comprehensive report of estimates on the `statsmodels` 
fashion, and anova(), to an ANOVA test for categorical covariates. Some other 
attributes may be helpful, such as mu, which returns a vector with expected 
values; pearson_residuals for the Pearson normalized residuals; aic, bic, 
and loglikelihood for model comparison.

Moreover, library `mcglm` provides methods to assist in specifying the 
matrix linear predictor through dependence matrices Z. There are three 
available methods: mc_id(), which crafts a matrix for independent 
realizations of outcome; mc_mixed(), which builds matrices for mixed 
models, and mc_ma() that build matrices for moving average fitting, 
popular models in time series analysis. The package `mcglm` of R 
language implements similar methods to aid in the matrix linear 
predictor specification. For in-depth details about those matrices, 
see @Bonat:2016.

The library can be installed in any Python environment that fulfills 
the requirements listed on PyPI Webpage. 

# Discussion

This article introduces the implementation of the McGLM framework in Python, 
providing a flexible statistical modeling approach for fitting a wide range 
of models. The `mcglm` library extends the `statsmodels` library and offers 
an accessible interface for accessing estimated parameters and goodness-of-fit 
measures.

As a new alternative for statistical analysis in Python, the `mcglm` 
library is poised for integration into the `statsmodels` library as a 
new API. Despite the limited assortment for covariance link function, the 
plurality of candidates for the other components enables many models listed 
in the statistical literature. Regarding software engineering, the library's 
object-oriented development philosophy provides a readable, extensible, 
self-contained, and testable implementation ideal for developing statistical 
models.

One significant difference between the `mcglm` library and existing R 
language packs is object-oriented development, suitable for statistical model 
production. While mixed models and copulas are similar statistical tools, the 
`mcglm` library's ability to analyze multiple responses sets it apart. We 
suggest conducting a comparative analysis of statistical tools and exploring 
different model specifications in the future. Integrating complementary models 
within `statsmodels` further solidifies Python's position as a conducive environment 
for conducting such analyses.

# References
