### Multivariate Covariance Generalized Linear Models

https://pypi.org/project/mcglm/

The mcglm package brings to python language one of the most powerful extensions to GLMs(Nelder, Wedderburn; 1972), the Multivariate Covariance Generalized Linear Models(Bonat, Jørgensen; 2016).

The GLMs have consolidated as a unified statistical model for analyzing non-gaussian independent data throughout the years. Notwithstanding enhancements to Linear Regression Models(Gauss), some key assumptions, such as the independence of components in the response, each element of the target belonging to an exponential dispersion family maintains.

MCGLM aims to expand the GLMs by allowing fitting on a wide variety of inner-dependent datasets, such as spatial and longitudinal, and supplant the exponential dispersion family output by second-moment assumptions(Wedderburn; 1974)

https://jeancmaia.github.io/posts/tutorial-mcglm/tutorial_mcglm.html

-----

The mcglm python package follows the standard pattern of the statsmodels library and aims to be another API on the package. Therefore, Python machine learning practitioners will be very familiar with this new statistical model. 


To install this package, use 

```bash
pip install mcglm
```

Tutorial MCGLM instills on the library usage by a wide-variety of examples(https://jeancmaia.github.io/posts/tutorial-mcglm/tutorial_mcglm.html). The following code snippet shows the model fitting for a Gaussian regression analysis.

```python
modelresults = MCGLM(endog=y, exog=X).fit()

modelresults.summary()
```


#### Workflow for developers/contributors

Contributions are key for the `mcglm` library to continue expanding. We need your help to make it a fabulous tool.

In order to submit new features or bug fixes, one must open a regular PR and ask for peer review. Currently, it is possible to include "jeancmaia" as a reviewer. Furthermore, developing the codebase aligned with the [PEP 8 style](https://peps.python.org/pep-0008/) and comprehensive unit tests are mandatory.

We recommend using the [poetry](https://python-poetry.org/) to create a local Python environment.

```
poetry install
```

Before pushing to GitHub, run the following commands:


1. To format the code base with black.

```
poetry run black mcglm
```

2. To run local tests to ensure realiablity of code.

```
poetry run python tests
```
