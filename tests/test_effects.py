import numpy as np

from analyzerct import effects


def test_unadjusted_effect_is_the_raw_difference():
    rng = np.random.default_rng(1)
    t = rng.integers(0, 2, 4000)
    y = (rng.random(4000) < 0.5 + 0.1 * t).astype(float)
    clusters = {"student": np.arange(4000) // 4, "class": np.arange(4000) // 100, "problem": np.arange(4000) % 50}
    res, _ = effects.analyze(y, t, None, clusters)
    raw = effects.raw_difference(y, t)
    assert np.allclose(res["effect (pp)"], raw["effect (pp)"])
    model = res.set_index("se_type").loc["model"]
    assert abs(model["SE (pp)"] - raw["SE (pp)"]) < 0.05


def test_a_good_control_narrows_the_ci():
    rng = np.random.default_rng(2)
    n = 4000
    t, x = rng.integers(0, 2, n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.3 * t + 2 * x)))).astype(float)
    clusters = {"student": np.arange(n), "class": np.arange(n), "problem": np.arange(n)}
    without, _ = effects.analyze(y, t, None, clusters, se_types=["model"])
    with_x, fit = effects.analyze(y, t, x, clusters, se_types=["model"])
    assert with_x["pp ± (half-width)"].iloc[0] < without["pp ± (half-width)"].iloc[0]
    assert fit["AUC (fixed effects)"] > 0.8
