import numpy as np

from vla_stress.analysis.stats import fit_dose_response, mcnemar_exact, wilson, x50_interp


def test_wilson_known_values():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.27 < hi < 0.28  # classic 0/10 -> [0, 0.278]
    lo, hi = wilson(10, 10)
    assert 0.72 < lo < 0.73 and hi == 1.0
    lo, hi = wilson(5, 10)
    assert abs((lo + hi) / 2 - 0.5) < 1e-9


def test_dose_response_recovers_threshold():
    rng = np.random.default_rng(0)
    x = np.repeat(np.linspace(0, 30, 7), 400)
    p = 0.9 / (1 + np.exp((x - 15) / 3))
    y = rng.random(x.size) < p
    fit = fit_dose_response(x, y)
    assert abs(fit["x50"] - 15) < 1.0
    assert abs(fit["s0"] - 0.9) < 0.05


def test_mcnemar_identical_is_not_significant():
    a = np.array([1, 0, 1, 1, 0])
    r = mcnemar_exact(a, a)
    assert r["p"] == 1.0 and r["a_only"] == 0
    r = mcnemar_exact(np.ones(12), np.zeros(12))
    assert r["p"] < 0.001


def test_x50_interp():
    x = np.repeat([0, 1, 2, 3], 10)
    y = np.concatenate([np.ones(8), np.zeros(2), np.ones(8), np.zeros(2), np.ones(2), np.zeros(8), np.zeros(10)])
    # rates 0.8, 0.8, 0.2, 0 -> half of 0.8 = 0.4 is crossed between 1 and 2, at 1 + 0.4/0.6
    assert abs(x50_interp(x, y) - (1 + 0.4 / 0.6)) < 1e-9
    assert x50_interp(x, np.ones(len(x))) == float("inf")
