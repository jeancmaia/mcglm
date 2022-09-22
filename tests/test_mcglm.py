import pytest
import numpy as np
import pandas as pd

from mcglm import __version__
from mcglm import MCGLM, MCGLMCAttributes, MCGLMMean, MCGLMVariance


def test_version():
    assert __version__ == "0.1.0"


TEST_X = np.array(
    [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10]]
)

TEST_BETA = [1, 2]


@pytest.fixture
def mcglmmean():
    return MCGLMMean()


@pytest.fixture
def mcglmvariance():
    return MCGLMVariance()


@pytest.fixture
def mcglmvattributes():
    return MCGLMCAttributes()


def test_an_invalid_link_function(mcglmmean):
    with pytest.raises(AssertionError):
        mcglmmean._link_function_attributes("TEST", "beta", "X")


def test_identity_link_function(mcglmmean):
    link = mcglmmean._link_function_attributes(
        link="identity", beta=TEST_BETA, X=TEST_X
    )

    assert np.array_equal(
        link.get("mu"), np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
    )
    assert np.array_equal(
        link.get("deriv"),
        np.array(
            [
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [1, 6],
                [1, 7],
                [1, 8],
                [1, 9],
                [1, 10],
            ]
        ),
    )


def test_logit_link_function(mcglmmean):
    link = mcglmmean._link_function_attributes(link="logit", beta=TEST_BETA, X=TEST_X)

    np.testing.assert_almost_equal(
        link.get("mu"),
        np.array(
            [
                0.9525741,
                0.9933071,
                0.9990889,
                0.9998766,
                0.9999833,
                0.9999977,
                0.9999997,
                1.0000000,
                1.0000000,
                1.0000000,
            ]
        ),
        decimal=7,
    )

    np.testing.assert_almost_equal(
        link.get("deriv"),
        np.array(
            [
                [0.045176659730912, 0.04517665973091],
                [0.006648056670790, 0.01329611334158],
                [0.000910221180122, 0.00273066354037],
                [0.000123379349765, 0.00049351739906],
                [0.000016701142911, 0.00008350571455],
                [0.000002260319189, 0.00001356191513],
                [0.000000305902133, 0.00000214131493],
                [0.000000041399374, 0.00000033119499],
                [0.000000005602796, 0.00000005042517],
                [0.000000000758256, 0.00000000758256],
            ]
        ),
        decimal=9,
    )


def test_log_link_function(mcglmmean):
    link = mcglmmean._link_function_attributes(link="log", beta=TEST_BETA, X=TEST_X)

    np.testing.assert_almost_equal(
        link.get("mu"),
        np.array(
            [
                20.08554,
                148.41316,
                1096.63316,
                8103.08393,
                59874.14172,
                442413.39201,
                3269017.37247,
                24154952.75358,
                178482300.96319,
                1318815734.48321,
            ]
        ),
        decimal=5,
    )

    np.testing.assert_almost_equal(
        link.get("deriv"),
        np.array(
            [
                [20.08554, 20.08554],
                [148.41316, 296.82632],
                [1096.63316, 3289.89948],
                [8103.08393, 32412.33571],
                [59874.14172, 299370.70858],
                [442413.39201, 2654480.35205],
                [3269017.37247, 22883121.60730],
                [24154952.75358, 193239622.02860],
                [178482300.96319, 1606340708.66869],
                [1318815734.48321, 13188157344.83215],
            ]
        ),
        decimal=4,
    )


def test_loglog_link_function(mcglmmean):
    link = mcglmmean._link_function_attributes(link="loglog", beta=TEST_BETA, X=TEST_X)

    np.testing.assert_almost_equal(
        link.get("mu"),
        np.array(
            [
                0.9514320,
                0.9932847,
                0.9990885,
                0.9998766,
                0.9999833,
                0.9999977,
                0.9999997,
                1.0000000,
                1.0000000,
                1.0000000,
            ]
        ),
        decimal=7,
    )

    np.testing.assert_almost_equal(
        link.get("deriv"),
        np.array(
            [
                [0.047369009677908, 0.04736900967791],
                [0.006692699677536, 0.01338539935507],
                [0.000911050815848, 0.00273315244754],
                [0.000123394575047, 0.00049357830019],
                [0.000016701421846, 0.00008350710923],
                [0.000002260324298, 0.00001356194579],
                [0.000000305902227, 0.00000214131559],
                [0.000000041399375, 0.00000033119500],
                [0.000000005602796, 0.00000005042517],
                [0.000000000758256, 0.00000000758256],
            ]
        ),
        decimal=9,
    )


TEST_VARIANCE_X = np.array([[1, -1.0], [1, -0.5], [1, 0.0], [1, 0.5], [1, 1.0]])

TEST_VARIANCE_BETA = [1, 0.5]


def test_power_variance_function(mcglmmean, mcglmvattributes):
    link = mcglmmean._link_function_attributes(
        link="logit", beta=TEST_VARIANCE_BETA, X=TEST_VARIANCE_X
    )

    variance = mcglmvattributes._generate_variance(
        variance_type="power", mu=link.get("mu"), power=1
    )

    np.testing.assert_almost_equal(
        variance.get("variance_sqrt_output"),
        np.array(
            [
                [0.7889609, 0, 0, 0, 0],
                [0, 0.824123, 0, 0, 0],
                [0, 0, 0.8550196, 0, 0],
                [0, 0, 0, 0.8816461, 0],
                [0, 0, 0, 0, 0.9041983],
            ]
        ),
        decimal=7,
    )

    np.testing.assert_almost_equal(
        variance.get("derivative_variance_sqrt_power"),
        np.array(
            [
                [-0.1870141, 0, 0, 0, 0],
                [0, -0.1594146, 0, 0, 0],
                [0, 0, -0.1339224, 0, 0],
                [0, 0, 0, -0.1110561, 0],
                [0, 0, 0, 0, -0.09105877],
            ]
        ),
        decimal=7,
    )

    np.testing.assert_almost_equal(
        variance.get("derivative_variance_sqrt_mu"),
        np.array([0.6337450, 0.6067056, 0.5847819, 0.5671210, 0.5529761]),
        decimal=7,
    )


def test_binomialp_variance_function(mcglmmean, mcglmvattributes):
    link = mcglmmean._link_function_attributes(
        link="logit", beta=TEST_VARIANCE_BETA, X=TEST_VARIANCE_X
    )

    variance = mcglmvattributes._generate_variance(
        variance_type="binomialP", mu=link.get("mu"), power=1
    )

    np.testing.assert_almost_equal(
        variance.get("variance_sqrt_output"),
        np.array(
            [
                [0.4847718, 0, 0, 0, 0],
                [0, 0.4667922, 0, 0, 0],
                [0, 0, 0.4434094, 0, 0],
                [0, 0, 0, 0.4160586, 0],
                [0, 0, 0, 0, 0.3861948],
            ]
        ),
        decimal=7,
    )

    np.testing.assert_almost_equal(
        variance.get("derivative_variance_sqrt_power"),
        np.array(
            [
                [-0.3510121, 0, 0, 0, 0],
                [0, -0.3556355, 0, 0, 0],
                [0, 0, -0.3606079, 0, 0],
                [0, 0, 0, -0.3648539, 0],
                [0, 0, 0, 0, -0.3674309],
            ]
        ),
        decimal=7,
    )

    np.testing.assert_almost_equal(
        variance.get("derivative_variance_sqrt_mu"),
        np.array([-0.2526123, -0.3838511, -0.5210953, -0.6664923, -0.8223167]),
        decimal=7,
    )


def test_binomialpq_variance_function(mcglmmean, mcglmvattributes):
    link = mcglmmean._link_function_attributes(
        link="logit", beta=TEST_VARIANCE_BETA, X=TEST_VARIANCE_X
    )

    variance = mcglmvattributes._generate_variance(
        variance_type="binomialPQ", mu=link.get("mu"), power=(1, 2)
    )

    np.testing.assert_almost_equal(
        variance.get("variance_sqrt_output"),
        np.array(
            [
                [0.2978648, 0, 0, 0, 0],
                [0, 0.2643962, 0, 0, 0],
                [0, 0, 0.2299502, 0, 0],
                [0, 0, 0, 0.1963427, 0],
                [0, 0, 0, 0, 0.1649488],
            ]
        ),
        decimal=7,
    )

    np.testing.assert_almost_equal(
        variance.get("derivative_variance_sqrt_p"),
        np.array(
            [
                [-0.07060543, 0, 0, 0, 0],
                [0, -0.05114361, 0, 0, 0],
                [0, 0, -0.03601729, 0, 0],
                [0, 0, 0, -0.02473222, 0],
                [0, 0, 0, 0, -0.01661144],
            ]
        ),
        decimal=7,
    )

    np.testing.assert_almost_equal(
        variance.get("derivative_variance_sqrt_q"),
        np.array(
            [
                [-0.1450716, 0, 0, 0, 0],
                [0, -0.1502922, 0, 0, 0],
                [0, 0, -0.1509924, 0, 0],
                [0, 0, 0, -0.1474464, 0],
                [0, 0, 0, 0, -0.1403231],
            ]
        ),
        decimal=7,
    )

    np.testing.assert_almost_equal(
        variance.get("derivative_variance_sqrt_mu"),
        np.array([-0.5496964, -0.6294789, -0.6977476, -0.7553482, -0.8033213]),
        decimal=7,
    )


TEST_Z = [
    np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    ),
]

TEST_TAU = [1, 0.8]

from mcglm.utils import diagonal, mc_matrix_linear_predictor


def test_mc_matrix_linear_predictor():
    linear_prediction = mc_matrix_linear_predictor(TEST_TAU, TEST_Z)

    np.testing.assert_almost_equal(
        linear_prediction,
        np.array(
            [
                [1.8, 0.8, 0.8, 0.8, 0.8],
                [0.8, 1.8, 0.8, 0.8, 0.8],
                [0.8, 0.8, 1.8, 0.8, 0.8],
                [0.8, 0.8, 0.8, 1.8, 0.8],
                [0.8, 0.8, 0.8, 0.8, 1.8],
            ]
        ),
        decimal=7,
    )


def test_mc_build_omega(mcglmvattributes):
    mcglmvattributes._z = [TEST_Z]
    mcglmvattributes._n_targets = 1
    omega = mcglmvattributes._generate_omega(tau=[TEST_TAU])

    np.testing.assert_almost_equal(
        omega[0],
        np.array(
            [
                [1.8, 0.8, 0.8, 0.8, 0.8],
                [0.8, 1.8, 0.8, 0.8, 0.8],
                [0.8, 0.8, 1.8, 0.8, 0.8],
                [0.8, 0.8, 0.8, 1.8, 0.8],
                [0.8, 0.8, 0.8, 0.8, 1.8],
            ]
        ),
        decimal=7,
    )


def test_mc_build_sigma__constant(mcglmvattributes):
    mcglmvattributes._z = [TEST_Z]
    mcglmvattributes._n_targets = 1
    omega = mcglmvattributes._generate_omega(tau=[TEST_TAU])

    sigma = mcglmvattributes._calculate_sigma(
        mu=np.array([1, 0.8]), power=2, omega=omega[0], variance="constant", Ntrial=1
    )
    np.testing.assert_almost_equal(
        sigma.get("sigma_chol"),
        np.array(
            [
                [1.3416408, 0.5962848, 0.5962848, 0.5962848, 0.5962848],
                [0.0, 1.2018504, 0.3698001, 0.3698001, 0.3698001],
                [0.0, 0.0, 1.1435437, 0.2690691, 0.2690691],
                [0.0, 0.0, 0.0, 1.1114379, 0.2117024],
                [0.0, 0.0, 0.0, 0.0, 1.0910895],
            ]
        ),
        decimal=7,
    )
    np.testing.assert_almost_equal(
        sigma.get("sigma_chol_inv"),
        np.array(
            [
                [0.7453560, -0.3698001, -0.2690691, -0.2117024, -0.1745743],
                [0.0, 0.8320503, -0.2690691, -0.2117024, -0.1745743],
                [0.0, 0.0, 0.8744746, -0.2117024, -0.1745743],
                [0.0, 0.0, 0.0, 0.8997354, -0.1745743],
                [0.0, 0.0, 0.0, 0.0, 0.9165151],
            ]
        ),
        decimal=7,
    )


def test_mc_build_sigma__tweedie_and_fixed_true(mcglmmean, mcglmvattributes):

    list_mu, mu, d = mcglmmean.calculate_mean_features(
        link=["log"], beta=[TEST_VARIANCE_BETA], X=[TEST_VARIANCE_X], offset=[None]
    )

    mcglmvattributes._z = [TEST_Z]
    mcglmvattributes._n_targets = 1
    omega = mcglmvattributes._generate_omega(tau=[TEST_TAU])

    sigma = mcglmvattributes._calculate_sigma(
        mu=list_mu[0], power=2, omega=omega[0], variance="tweedie", Ntrial=[1]
    )
    np.testing.assert_almost_equal(
        sigma.get("sigma_chol"),
        np.array(
            [
                [2.2119917, 1.2623349, 1.6208701, 2.0812384, 2.6723630],
                [0.0, 2.5443174, 1.0052210, 1.2907293, 1.6573292],
                [0.0, 0.0, 3.1084742, 0.9391435, 1.2058841],
                [0.0, 0.0, 0.0, 3.8792993, 0.9487846],
                [0.0, 0.0, 0.0, 0.0, 4.8899237],
            ]
        ),
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sigma.get("sigma_chol_inv"),
        np.array(
            [
                [0.45208126, -0.22429512, -0.16319867, -0.12840403, -0.10588467],
                [0.0, 0.39303273, -0.12709925, -0.10000116, -0.08246307],
                [0.0, 0.0, 0.32170124, -0.07788098, -0.06422230],
                [0.0, 0.0, 0.0, 0.25777851, -0.05001638],
                [0.0, 0.0, 0.0, 0.0, 0.20450217],
            ]
        ),
        decimal=6,
    )

    sigma_derivatives = mcglmvattributes._calculate_sigma_derivatives(
        mu=list_mu[0],
        power=2,
        variance="tweedie",
        z=TEST_Z,
        power_fixed=True,
        Ntrial=[1],
        omegas=omega,
    )
    np.testing.assert_almost_equal(
        sigma_derivatives.get("sigma_derivative")[0],
        np.array(
            [
                [2.71828183, 0.0, 0.0, 0, 0],
                [0.0, 4.48168907, 0, 0, 0],
                [0.0, 0.0, 7.3890561, 0.0, 0.0],
                [0.0, 0.0, 0.0, 12.18249396, 0.0],
                [0.0, 0.0, 0.0, 0.0, 20.08553692],
            ]
        ),
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sigma_derivatives.get("sigma_derivative")[1],
        np.array(
            [
                [2.718282, 3.490343, 4.481689, 5.754603, 7.389056],
                [3.490343, 4.481689, 5.754603, 7.389056, 9.487736],
                [4.481689, 5.754603, 7.389056, 9.487736, 12.182494],
                [5.754603, 7.389056, 9.487736, 12.182494, 15.642632],
                [7.389056, 9.487736, 12.182494, 15.642632, 20.085537],
            ]
        ),
        decimal=6,
    )

    sigma_derivatives = mcglmvattributes._calculate_sigma_derivatives(
        mu=list_mu[0],
        power=1,
        variance="tweedie",
        z=TEST_Z,
        power_fixed=False,
        Ntrial=[1],
        omegas=omega,
    )
    np.testing.assert_almost_equal(
        sigma_derivatives.get("sigma_derivative")[0][0],
        np.array(
            [
                [1.4838, 0.934, 1.2702, 1.679, 2.174],
            ]
        ),
        decimal=3,
    )
    np.testing.assert_almost_equal(
        sigma_derivatives.get("sigma_derivative")[0][1],
        np.array(
            [
                [0.9341, 2.857, 1.6792, 2.174, 2.772],
            ]
        ),
        decimal=3,
    )
    np.testing.assert_almost_equal(
        sigma_derivatives.get("sigma_derivative")[0][2],
        np.array(
            [
                [1.2702, 1.679, 4.8929, 2.772, 3.490],
            ]
        ),
        decimal=3,
    )
    np.testing.assert_almost_equal(
        sigma_derivatives.get("sigma_derivative")[0][3],
        np.array(
            [
                [1.6792, 2.174, 2.7721, 7.853, 4.350],
            ]
        ),
        decimal=3,
    )
    np.testing.assert_almost_equal(
        sigma_derivatives.get("sigma_derivative")[0][4],
        np.array(
            [
                [2.1746, 2.772, 3.4903, 4.350, 12.100],
            ]
        ),
        decimal=3,
    )


def test_apply_build_sigma_on_list(mcglmmean, mcglmvattributes):

    list_mu, mu, d = mcglmmean.calculate_mean_features(
        link=["log", "log"],
        beta=[[3, 2.5], [0.2, 1.1]],
        X=[TEST_VARIANCE_X, TEST_VARIANCE_X],
        offset=[None, None],
    )

    mcglmvattributes._z = [TEST_Z, TEST_Z]
    mcglmvattributes._n_targets = 2
    mcglmvattributes._n_obs = 5
    mcglmvattributes._ntrial = [None, None]
    mcglmvattributes._variance = ["constant", "constant"]

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
    ) = mcglmvattributes.c_inverse(
        mu=list_mu, power=[3, 2], rho=2, tau=[[1, 0.8], [2, 1.8]], full_response=True
    )

    np.testing.assert_almost_equal(
        sigma_chol[0],
        np.array(
            [
                [1.3416408, 0.5962848, 0.5962848, 0.5962848, 0.5962848],
                [0.0, 1.2018504, 0.3698001, 0.3698001, 0.3698001],
                [0.0, 0.0, 1.1435437, 0.2690691, 0.2690691],
                [0.0, 0.0, 0.0, 1.1114379, 0.2117024],
                [0.0, 0.0, 0.0, 0.0, 1.0910895],
            ]
        ),
        decimal=5,
    )

    np.testing.assert_almost_equal(
        sigma_chol[1],
        np.array(
            [
                [1.9493589, 0.9233805, 0.9233805, 0.9233805, 0.9233805],
                [0.0, 1.7167902, 0.5518254, 0.5518254, 0.5518254],
                [0.0, 0.0, 1.6256867, 0.3954373, 0.3954373],
                [0.0, 0.0, 0.0, 1.5768597, 0.3085160],
                [0.0, 0.0, 0.0, 0.0, 1.5463843],
            ]
        ),
        decimal=5,
    )

    np.testing.assert_almost_equal(
        sigma_chol_inv[0],
        np.array(
            [
                [0.7453560, -0.3698001, -0.2690691, -0.2117024, -0.1745743],
                [0.0, 0.8320503, -0.2690691, -0.2117024, -0.1745743],
                [0.0, 0.0, 0.8744746, -0.2117024, -0.1745743],
                [0.0, 0.0, 0.0, 0.8997354, -0.1745743],
                [0.0, 0.0, 0.0, 0.0, 0.9165151],
            ]
        ),
        decimal=5,
    )

    np.testing.assert_almost_equal(
        sigma_chol_inv[1],
        np.array(
            [
                [0.5129892, -0.2759127, -0.1977186, -0.1542580, -0.1265224],
                [0.0, 0.5824824, -0.1977186, -0.1542580, -0.1265224],
                [0.0, 0.0, 0.6151247, -0.1542580, -0.1265224],
                [0.0, 0.0, 0.0, 0.6341718, -0.1265224],
                [0.0, 0.0, 0.0, 0.0, 0.6466698],
            ]
        ),
        decimal=5,
    )


def test_mc_derivative_sigma_between(mcglmvattributes):
    sigma_two = mcglmvattributes._mc_derivative_sigma_between(2)

    np.testing.assert_almost_equal(
        sigma_two,
        [np.array([[0, 1], [1, 0]])],
        decimal=5,
    )

    sigma_three = mcglmvattributes._mc_derivative_sigma_between(3)

    np.testing.assert_almost_equal(
        sigma_three,
        [
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        ],
        decimal=5,
    )


def test_mc_build_sigma_between(mcglmvattributes):

    sigmab, _ = mcglmvattributes._sigma_between_values(0, 15)

    np.testing.assert_almost_equal(
        sigmab,
        np.array(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            ]
        ),
        decimal=5,
    )


def test_mc_build_c_beta_and_c_equal_tofalse(mcglmmean, mcglmvattributes):

    list_mu, mu, d = mcglmmean.calculate_mean_features(
        link=["log", "log"],
        beta=[[3, 2.5], [0.2, 1.1]],
        X=[TEST_VARIANCE_X, TEST_VARIANCE_X],
        offset=[None, None],
    )

    mcglmvattributes._z = [TEST_Z, TEST_Z]
    mcglmvattributes._n_targets = 2
    mcglmvattributes._n_obs = 5
    mcglmvattributes._ntrial = [None, None]
    mcglmvattributes._variance = ["constant", "constant"]
    mcglmvattributes._power_fixed = [False, True]

    # mu, power, rho, tau

    c_inverse, c_derivatives, c_values = mcglmvattributes.c_complete(
        mu=mu, power=[3, 2], rho=2, tau=[[1, 0.8], [2, 1.8]]
    )

    np.testing.assert_almost_equal(
        c_derivatives[0],
        np.array(
            [
                [0, 0, 0, 0, 0, 2.615339, 1.238845, 1.238845, 1.238845, 1.238845],
                [0, 0, 0, 0, 0, 1.162373, 2.613923, 1.213809, 1.213809, 1.213809],
                [0, 0, 0, 0, 0, 1.162373, 1.185467, 2.613707, 1.206863, 1.206863],
                [0, 0, 0, 0, 0, 1.162373, 1.185467, 1.192085, 2.613644, 1.203959],
                [0, 0, 0, 0, 0, 1.162373, 1.185467, 1.192085, 1.194888, 2.613620],
                [2.615339, 1.162373, 1.162373, 1.162373, 1.162373, 0, 0, 0, 0, 0],
                [1.238845, 2.613923, 1.185467, 1.185467, 1.185467, 0, 0, 0, 0, 0],
                [1.238845, 1.213809, 2.613707, 1.192085, 1.192085, 0, 0, 0, 0, 0],
                [1.238845, 1.213809, 1.206863, 2.613644, 1.194888, 0, 0, 0, 0, 0],
                [1.238845, 1.213809, 1.206863, 1.203959, 2.613620, 0, 0, 0, 0, 0],
            ]
        ),
        decimal=6,
    )

    np.testing.assert_almost_equal(
        c_derivatives[1][9],
        np.array(
            [
                7.745518e-01,
                4.803566e-01,
                3.495108e-01,
                2.749936e-01,
                1.417285e00,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
        decimal=6,
    )
    np.testing.assert_almost_equal(
        c_derivatives[1][8],
        np.array(
            [
                6.434631e-01,
                3.990589e-01,
                2.903580e-01,
                1.420208e00,
                -2.677988e-01,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
        decimal=6,
    )
    np.testing.assert_almost_equal(
        c_derivatives[1][7],
        np.array(
            [
                5.145533e-01,
                3.191124e-01,
                1.426351e00,
                -2.756085e-01,
                -3.432484e-01,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
        decimal=6,
    )
    np.testing.assert_almost_equal(
        c_derivatives[1][6],
        np.array(
            [
                3.904668e-01,
                1.442060e00,
                -2.868948e-01,
                -3.846066e-01,
                -4.789967e-01,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
        decimal=6,
    )
    np.testing.assert_almost_equal(
        c_derivatives[1][5],
        np.array(
            [
                1.501908e00,
                -3.046638e-01,
                -4.800668e-01,
                -6.435699e-01,
                -8.015149e-01,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
        decimal=6,
    )
    np.testing.assert_almost_equal(
        c_derivatives[1][4],
        np.array(
            [
                0.000000e00,
                2.775558e-17,
                1.387779e-17,
                0.000000e00,
                1.000000e00,
                -0.8015149,
                -0.4789967,
                -0.3432484,
                -0.2677988,
                1.4172846,
            ]
        ),
        decimal=4,
    )
    np.testing.assert_almost_equal(
        c_derivatives[1][3],
        np.array(
            [
                -5.551115e-17,
                -5.551115e-17,
                -4.163336e-17,
                1.000000e00,
                0.000000e00,
                -0.6435699,
                -0.3846066,
                -0.2756085,
                1.4202079,
                0.2749936,
            ]
        ),
        decimal=6,
    )
    np.testing.assert_almost_equal(
        c_derivatives[1][2],
        np.array(
            [
                0.000000e00,
                1.387779e-17,
                1.000000e00,
                -4.163336e-17,
                1.387779e-17,
                -0.4800668,
                -0.2868948,
                1.4263506,
                0.2903580,
                0.3495108,
            ]
        ),
        decimal=6,
    )
    np.testing.assert_almost_equal(
        c_derivatives[1][1],
        np.array(
            [
                -5.551115e-17,
                1.000000e00,
                1.387779e-17,
                -5.551115e-17,
                2.775558e-17,
                -0.3046638,
                1.4420601,
                0.3191124,
                0.3990589,
                0.4803566,
            ]
        ),
        decimal=6,
    )
    np.testing.assert_almost_equal(
        c_derivatives[1][0],
        np.array(
            [
                1.000000e00,
                -5.551115e-17,
                0.000000e00,
                -5.551115e-17,
                0.000000e00,
                1.5019083,
                0.3904668,
                0.5145533,
                0.6434631,
                0.7745518,
            ]
        ),
        decimal=6,
    )


def test_mc_quasi_score(mcglmmean):

    x1 = np.linspace(-5, 5, 15)
    x2 = np.linspace(45, 80, 15)
    X = np.ones((15, 3))

    X[:, 1] = x1
    X[:, 2] = x2

    y = np.linspace(45, 80, 15)

    Z0 = diagonal(15, X[:, 0].copy())
    Z1 = np.ones((15, 15))
    Z = [Z0, Z1]

    W = diagonal(15, X[:, 0].copy())

    mcglmmean._z = [Z]
    mcglmmean._n_targets = 1
    mcglmmean._n_obs = 15
    mcglmmean._ntrial = [None]
    mcglmmean._link = ["log"]
    mcglmmean._variance = ["constant"]
    mcglmmean._offset = [None]
    mcglmmean._X = [X]
    mcglmmean._y_values = y
    mcglmmean._y_names = ["output"]

    # beta, W, power, rho, tau
    (new_beta, score, sensitivity, variability) = mcglmmean.update_beta(
        beta=[[0.8, 0.05, 0]], W=W, power=[3], rho=1, tau=[[1, 5]]
    )

    np.testing.assert_almost_equal(
        score,
        np.array([81.01208324, 1112.53370168, 8957.12315859]),
        decimal=0,
    )

    np.testing.assert_almost_equal(
        sensitivity,
        np.array(
            [
                [-2.80369448e00, -3.72056908e01, -3.05450823e02],
                [-3.72056908e01, -7.52022067e02, -4.95743291e03],
                [-3.05450823e02, -4.95743291e03, -3.64416916e04],
            ]
        ),
        decimal=0,
    )

    np.testing.assert_almost_equal(
        variability,
        np.array(
            [
                [2.80369448e00, 3.72056908e01, 3.05450823e02],
                [3.72056908e01, 7.52022067e02, 4.95743291e03],
                [3.05450823e02, 4.95743291e03, 3.64416916e04],
            ]
        ),
        decimal=1,
    )


def test_mc_pearson(mcglmmean, mcglmvariance):

    x1 = np.linspace(-5, 5, 15)
    x2 = np.linspace(45, 80, 15)
    X = np.ones((15, 3))

    X[:, 1] = x1
    X[:, 2] = x2

    y = np.linspace(45, 80, 15)

    Z0 = diagonal(15, X[:, 0].copy())
    Z1 = np.ones((15, 15))
    Z = [Z0, Z1]

    W = diagonal(15, X[:, 0].copy())

    mcglmmean._z = [Z]
    mcglmmean._n_targets = 1
    mcglmmean._n_obs = 15
    mcglmmean._ntrial = [None]
    mcglmmean._link = ["log"]
    mcglmmean._variance = ["constant"]
    mcglmmean._offset = [None]
    mcglmmean._X = [X]
    mcglmmean._y_values = y
    mcglmmean._y_names = ["output"]

    mu_attributes_per_response, mu, _ = mcglmmean.calculate_mean_features(
        link=["log"], beta=[[0, 1, 0]], X=[X], offset=[None]
    )

    mcglmvariance._z = [Z]
    mcglmvariance._n_targets = 1
    mcglmvariance._n_obs = 15
    mcglmvariance._ntrial = [None]
    mcglmvariance._link = ["log"]
    mcglmvariance._variance = ["constant"]
    mcglmvariance._offset = [None]
    mcglmvariance._X = [X]
    mcglmvariance._y_values = y
    mcglmvariance._y_names = ["output"]
    mcglmvariance._power_fixed = [True]
    mcglmvariance._tuning = 0.5

    (
        new_dispersion,
        c_inverse,
        c_values,
        c_derivatives_componentes,
        var_sensitivity,
    ) = mcglmvariance.update_covariates(
        mu_attributes=mu_attributes_per_response,
        rho=1,
        power=[3],
        tau=[[1, 5]],
        W=W,
        dispersion=[0, 0],
        mu=mu,
    )

    np.testing.assert_almost_equal(
        var_sensitivity,
        np.array([[-14.000173130, -0.002596953], [-0.002596953, -0.038954294]]),
        decimal=5,
    )

    np.testing.assert_almost_equal(
        c_derivatives_componentes[0],
        np.array(
            [
                [
                    0.93421053,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                ],
                [
                    -0.06578947,
                    0.93421053,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                ],
                [
                    -0.06578947,
                    -0.06578947,
                    0.93421053,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                ],
                [
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    0.93421053,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                ],
                [
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    0.93421053,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                ],
                [
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    0.93421053,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                ],
                [
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    0.93421053,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                ],
                [
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    0.93421053,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                ],
                [
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    0.93421053,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                ],
                [
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    0.93421053,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                ],
                [
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    0.93421053,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                ],
                [
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    0.93421053,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                ],
                [
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    0.93421053,
                    -0.06578947,
                    -0.06578947,
                ],
                [
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    0.93421053,
                    -0.06578947,
                ],
                [
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    -0.06578947,
                    0.93421053,
                ],
            ]
        ),
        decimal=5,
    )

    np.testing.assert_almost_equal(
        c_derivatives_componentes[1],
        np.array(
            [
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
                [
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                    0.01315789,
                ],
            ]
        ),
        decimal=7,
    )


def test_fit_mcglm__link_tweedie_and_power_fixed__one_variable():

    data = pd.read_csv("tests/output_third_test.csv")

    x1 = data["x1"].values
    x2 = data["x2"].values
    X = np.ones((100, 3))

    X[:, 1] = x1
    X[:, 2] = x2

    y = data["y"]

    Z0 = diagonal(100, X[:, 0].copy())
    Z = [Z0]

    list_link = "log"
    list_Z = Z
    list_X = pd.DataFrame(X)
    list_Y = y
    list_variance = "tweedie"
    list_power_fixed = True

    mdl = MCGLM(
        endog=list_Y,
        exog=list_X,
        z=list_Z,
        link=list_link,
        variance=list_variance,
        power_fixed=list_power_fixed,
        tuning=1,
    )

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
    ) = mdl._fit()
    np.testing.assert_almost_equal(
        regression_historical[-1], np.array([[4.97, 1.93, -1.12]]), decimal=2
    )
    np.testing.assert_almost_equal(
        dispersion_historical[-1],
        np.array([256.11]),
        decimal=2,
    )


def test_fit_mcglm__link_tweedie_and_power_fixed__two_variable():

    data = pd.read_csv("tests/output_fourth_test.csv")
    x1 = data["x1"].values
    x2 = data["x2"].values
    X = np.ones((100, 3))

    X[:, 1] = x1
    X[:, 2] = x2

    y1 = data["y1"]
    y2 = data["y2"]

    Z0 = diagonal(100, X[:, 0].copy())
    Z = [Z0]

    list_link = ["log", "log"]
    list_Z = [Z, Z]
    list_X = [pd.DataFrame(X), pd.DataFrame(X)]
    list_Y = [y1, y2]
    list_variance = ["tweedie", "tweedie"]
    list_power_fixed = [True, True]
    list_Ntrial = [None, None]

    mdl = MCGLM(
        endog=list_Y,
        exog=list_X,
        z=list_Z,
        link=list_link,
        variance=list_variance,
        power_fixed=list_power_fixed,
        ntrial=list_Ntrial,
        tuning=1,
    )
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
    ) = mdl._fit()

    np.testing.assert_almost_equal(
        regression_historical[-1],
        np.array(
            [[4.9733212, 1.9326481, -1.1163137], [-0.8786206, 1.9661887, -0.1196820]]
        ),
        decimal=2,
    )
    np.testing.assert_almost_equal(
        np.array(dispersion_historical[-1]),
        np.array([-6.95361666e-02, 2.5610173e02, 1.07966296e00]),
        decimal=1,
    )

    # mdl_report.summary()


def test_fit_mcglm__simulation_two_labels():

    import patsy

    u = np.array([5, 10, 15, 20, 30, 40, 60, 80, 100])
    lot1 = np.array([118, 58, 42, 35, 27, 25, 21, 19, 18])
    lot2 = np.array([69, 35, 26, 21, 18, 16, 13, 12, 12])

    data = pd.DataFrame([], columns=["u", "lot1", "lot2"])
    data["u"] = u
    data["lot1"] = lot1
    data["lot2"] = lot2

    data["log_u"] = np.log(data["u"])

    Y1, X1 = patsy.dmatrices("lot1 ~ log_u", data, return_type="dataframe")
    Y2, X2 = patsy.dmatrices("lot2 ~ log_u", data, return_type="dataframe")

    list_link = ["log", "log"]
    list_Z = [[diagonal(9, np.ones(9))], [diagonal(9, np.ones(9))]]
    list_X = [X1, X2]
    list_Y = [Y1["lot1"], Y2["lot2"]]
    list_variance = ["tweedie", "tweedie"]
    list_power_fixed = [True, True]
    list_Ntrial = [None, None]

    mdl = MCGLM(
        endog=list_Y,
        exog=list_X,
        z=list_Z,
        link=list_link,
        variance=list_variance,
        power_fixed=list_power_fixed,
        ntrial=list_Ntrial,
        tuning=0.5,
        tol=0.001,
    )
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
    ) = mdl._fit()

    np.testing.assert_almost_equal(
        regression_historical[-1],
        np.array(
            [
                [5.74296, -0.6818],
                [5.1438, -0.6422],
            ]
        ),
        decimal=1,
    )
    np.testing.assert_almost_equal(
        np.array(dispersion_historical[-1]),
        np.array([0.98070244, 0.799, 0.464]),
        decimal=1,
    )

    mdlreport = mdl.fit()
    mdlreport.summary()


def test_fit_mcglm__simulation_two_labels_binomial():

    import patsy

    u = np.array([5, 10, 15, 20, 30, 40, 60, 80, 100])
    lot1 = np.array([118, 58, 42, 35, 27, 25, 21, 19, 18])
    lot2 = np.array([69, 35, 26, 21, 18, 16, 13, 12, 12])

    data = pd.DataFrame([], columns=["u", "lot1", "lot2"])
    data["u"] = u
    data["lot1"] = lot1 / 118
    data["lot2"] = lot2 / 69

    data["log_u"] = np.log(data["u"])

    Y1, X1 = patsy.dmatrices("lot1 ~ log_u", data, return_type="dataframe")
    Y2, X2 = patsy.dmatrices("lot2 ~ log_u", data, return_type="dataframe")

    list_link = ["probit", "loglog", "logit"]
    list_Z = [
        [diagonal(9, np.ones(9))],
        [diagonal(9, np.ones(9))],
        [diagonal(9, np.ones(9))],
    ]
    list_X = [X1, X2, X1]
    list_Y = [Y1["lot1"], Y2["lot2"], Y1["lot1"]]
    list_variance = ["binomialP", "binomialP", "binomialPQ"]
    list_power_fixed = [True, True, True]
    list_Ntrial = [1, 1, 1]

    mdl = MCGLM(
        endog=list_Y,
        exog=list_X,
        z=list_Z,
        link=list_link,
        variance=list_variance,
        power_fixed=list_power_fixed,
        ntrial=list_Ntrial,
        tuning=1,
        tol=0.001,
    )
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
    ) = mdl._fit()

    np.testing.assert_almost_equal(
        regression_historical[-1],
        np.array(
            [
                [-0.14156894, -0.14070916],
                [0.232246, -0.13404584],
                [-0.19223043, -0.24001806],
            ]
        ),
        decimal=2,
    )
    np.testing.assert_almost_equal(
        np.array(dispersion_historical[-1]),
        np.array(
            [0.99947835, 0.99999981, 0.99947562, 0.2262965, 0.20997098, 0.22344229]
        ),
        decimal=2,
    )
