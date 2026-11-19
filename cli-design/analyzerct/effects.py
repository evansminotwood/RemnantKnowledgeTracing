"""Treatment-effect models for the RCT: y ~ treatment * covariate (+ optional problem fixed effects), logistic.

For each covariate it reports
* the treatment odds ratio at the covariate's mean (covariate is centered), with its CI,
* the treatment effect in percentage points by g-computation (predict every log as treated and as control,
  average the difference), with a delta-method CI,
* fit: AUC of the linear predictor and latent-scale R² = var(eta) / (var(eta) + pi^2 / 3), which is the
  same formula as Nakagawa's marginal R² for a model without random effects,
under several standard-error types, so their CIs can be compared directly:
  model           inverse information (assumes independent logs)
  hc0             heteroskedasticity-robust sandwich
  student/class/problem   cluster-robust sandwich, one-way (CR1: G/(G-1) * (n-1)/(n-k) correction)
  student+problem two-way cluster-robust (Cameron-Gelbach-Miller): V_student + V_problem - V_student×problem
"""
import numpy as np
import pandas as pd

SE_TYPES = ["model", "hc0", "student", "class", "problem", "student+problem"]
Z = 1.959963984540054


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit_logistic(X, y, ridge=None, tol=1e-10, max_iter=100):
    """Newton / IRLS maximum likelihood. Returns coefficients and the bread (X'WX + ridge)^-1.

    ridge: optional per-coefficient L2 penalty; used for problem fixed effects, which run off to infinity for
    problems where every outcome is the same (separation)."""
    pen = np.zeros(X.shape[1]) if ridge is None else ridge
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = _sigmoid(X @ beta)
        w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + np.diag(pen)
        step = np.linalg.solve(H, X.T @ (y - p) - pen * beta)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    p = _sigmoid(X @ beta)
    H = X.T @ (X * (p * (1 - p))[:, None]) + np.diag(pen)
    return beta, np.linalg.inv(H)


def _meat(scores, groups):
    codes = pd.factorize(groups)[0]
    U = np.zeros((codes.max() + 1, scores.shape[1]))
    np.add.at(U, codes, scores)
    return U.T @ U, codes.max() + 1


def vcov(kind, X, y, beta, bread, clusters):
    n, k = X.shape
    scores = X * (y - _sigmoid(X @ beta))[:, None]
    if kind == "model":
        return bread
    if kind == "hc0":
        return bread @ (scores.T @ scores) @ bread

    def one_way(g):
        M, G = _meat(scores, g)
        return (G / (G - 1)) * ((n - 1) / (n - k)) * (bread @ M @ bread)

    if kind in ("student", "class", "problem"):
        return one_way(clusters[kind])
    if kind == "student+problem":
        both = pd.Series(clusters["student"]).astype(str) + "|" + pd.Series(clusters["problem"]).astype(str)
        V = one_way(clusters["student"]) + one_way(clusters["problem"]) - one_way(both.to_numpy())
        w, Q = np.linalg.eigh((V + V.T) / 2)   # the difference can be slightly non-PSD
        return (Q * np.clip(w, 0, None)) @ Q.T
    raise ValueError(kind)


def design(t, x, fe=None):
    """[intercept, treatment, covariate, treatment×covariate (+ problem dummies)]; x=None -> unadjusted."""
    cols = [np.ones_like(t, dtype=float), t.astype(float)]
    names = ["intercept", "treatment"]
    if x is not None:
        cols += [x, t * x]
        names += ["covariate", "treatment:covariate"]
    X = np.column_stack(cols)
    if fe is not None:
        X = np.hstack([X, fe])
        names += [f"fe{i}" for i in range(fe.shape[1])]
    return X, names


def problem_dummies(problem):
    codes, _ = pd.factorize(problem)
    D = np.zeros((len(problem), codes.max() + 1))
    D[np.arange(len(problem)), codes] = 1.0
    return D[:, 1:]   # drop one level (the intercept carries it)


def analyze(y, t, x, clusters, fe=None, se_types=SE_TYPES, fe_ridge=1.0):
    """Fit one model and return a long table (one row per SE type) plus fit statistics."""
    from sklearn.metrics import roc_auc_score
    xc = None if x is None else x - x.mean()   # treatment OR at the covariate mean
    X, names = design(t, xc, fe)
    ridge = None
    if fe is not None:   # light penalty on the problem effects only
        ridge = np.zeros(X.shape[1])
        ridge[X.shape[1] - fe.shape[1]:] = fe_ridge
    beta, bread = fit_logistic(X, y, ridge)
    eta = X @ beta
    fitstats = {"n": len(y), "AUC (fixed effects)": roc_auc_score(y, eta),
                "latent R2": np.var(eta) / (np.var(eta) + np.pi ** 2 / 3),
                "log-likelihood": float(np.sum(y * eta - np.log1p(np.exp(eta))))}

    # g-computation: everyone treated vs everyone control
    X1, _ = design(np.ones_like(t), xc, fe)
    X0, _ = design(np.zeros_like(t), xc, fe)
    p1, p0 = _sigmoid(X1 @ beta), _sigmoid(X0 @ beta)
    rd = np.mean(p1 - p0)
    grad = (X1 * (p1 * (1 - p1))[:, None]).mean(0) - (X0 * (p0 * (1 - p0))[:, None]).mean(0)

    rows = []
    for kind in se_types:
        V = vcov(kind, X, y, beta, bread, clusters)
        se_b = np.sqrt(V[1, 1])
        se_rd = float(np.sqrt(grad @ V @ grad))
        row = {"se_type": kind,
               "treatment OR": np.exp(beta[1]), "OR lo": np.exp(beta[1] - Z * se_b), "OR hi": np.exp(beta[1] + Z * se_b),
               "OR CI width": np.exp(beta[1] + Z * se_b) - np.exp(beta[1] - Z * se_b),
               "OR p": 2 * (1 - _norm_cdf(abs(beta[1] / se_b))),
               "effect (pp)": 100 * rd, "SE (pp)": 100 * se_rd,
               "pp lo": 100 * (rd - Z * se_rd), "pp hi": 100 * (rd + Z * se_rd),
               "pp ± (half-width)": 100 * Z * se_rd, "pp CI width": 100 * 2 * Z * se_rd,
               "pp p": 2 * (1 - _norm_cdf(abs(rd / se_rd)))}
        if x is not None:
            se_i = np.sqrt(V[3, 3])
            row |= {"interaction OR": np.exp(beta[3]), "int lo": np.exp(beta[3] - Z * se_i),
                    "int hi": np.exp(beta[3] + Z * se_i), "int p": 2 * (1 - _norm_cdf(abs(beta[3] / se_i)))}
        rows.append(row)
    return pd.DataFrame(rows), fitstats


def _norm_cdf(z):
    from math import erf, sqrt
    return 0.5 * (1 + erf(z / sqrt(2)))


def raw_difference(y, t):
    """Difference in proportions with the textbook SE, as a reference point for every CI above."""
    p1, p0 = y[t == 1].mean(), y[t == 0].mean()
    n1, n0 = (t == 1).sum(), (t == 0).sum()
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return {"effect (pp)": 100 * (p1 - p0), "SE (pp)": 100 * se, "pp lo": 100 * (p1 - p0 - Z * se),
            "pp hi": 100 * (p1 - p0 + Z * se), "pp ± (half-width)": 100 * Z * se, "pp CI width": 100 * 2 * Z * se,
            "treated rate": p1, "control rate": p0, "n treated": int(n1), "n control": int(n0)}
