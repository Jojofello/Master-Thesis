import numpy as np
from scipy.stats import invgamma

def check_stab(a, p):
    """
    ---------------------------------------------------------------------
    FUNCTION: check_stab
    ---------------------------------------------------------------------
    Test stationarity of an AR(p) with coefficients `a` by inspecting
    the spectral radius of its companion matrix.

    INPUTS
        a : 1-D numpy.ndarray of length >= p
            Coefficients [phi_1, phi_2, ..., phi_p, ...]. Only the first
            p entries are used; later entries (e.g. additional
            regressors) are ignored.
        p : int (non-negative)
            AR order. p = 0 is treated as trivially stable.

    OUTPUT
        ok : bool
             True if all roots of the AR polynomial lie strictly inside
             the unit circle (|eigenvalue(C)| < 1 for every eigenvalue),
             False otherwise.
    ---------------------------------------------------------------------
    """
    if p == 0:
        return True
    C = np.zeros((p, p))
    C[0, :] = a.flatten()[:p]
    if p > 1:
        C[1:, :-1] = np.eye(p - 1)
    return np.all(np.abs(np.linalg.eigvals(C)) < 1.0)

def flat_prior(Y, X, nx, q, p, pr, own_lag_mean=0.0):
    """
    ---------------------------------------------------------------------
    FUNCTION: flat_prior
    ---------------------------------------------------------------------
    Return the precision-matrix / mean pair (Hm, bm) that encodes a flat
    Normal prior on the regression coefficient vector. With the flat
    prior used in this script `Hm = 0` and `bm = 0`, so the posterior
    conditional for beta collapses to the standard Bayesian linear
    regression result.

    The signature matches the more elaborate Minnesota-prior helper
    used elsewhere in the project so that the two can be swapped without
    touching the calling code.

    INPUTS
        Y            : numpy.ndarray of shape (T,)
                       Dependent variable (unused in the flat case).
        X            : numpy.ndarray of shape (T, nc)
                       Regressor matrix (only nc = X.shape[1] is needed).
        nx           : int
                       Number of contemporaneous shock-decomposition
                       regressors per period (here always 2: +/- parts).
        q            : int
                       Number of lags of the exogenous shock (unused).
        p            : int
                       AR lag length on the dependent variable (unused).
        pr           : float
                       Prior tightness placeholder (unused).
        own_lag_mean : float, default 0.0
                       Prior mean for the own-lag coefficients (unused).

    OUTPUT
        Hm : numpy.ndarray of shape (nc, nc)
             Prior precision matrix.  All zeros for a flat prior.
        bm : numpy.ndarray of shape (nc,)
             Prior mean.            All zeros for a flat prior.
    ---------------------------------------------------------------------
    """
    nc = X.shape[1]
    Hm = np.zeros((nc, nc))
    bm = np.zeros(nc)
    return Hm, bm

def run_irf_constvar(dy, dz, p, q, Reps, burn, Nirf, prior_t, own_lag_mean=0.0):
    """
    ---------------------------------------------------------------------
    FUNCTION: run_irf_constvar
    ---------------------------------------------------------------------
    Run a Gibbs sampler for the constant-variance ADL model and produce
    posterior IRF summaries (median + 16/84th percentiles) for both a
    positive and a negative one-unit shock.

    INPUTS
        dy           : 1-D numpy.ndarray of length Tr
                       Dependent variable y_t (e.g. EUA or ESG returns).
        dz           : 1-D numpy.ndarray of length Tr
                       Shock series z_t. It is split into z_t^+ and z_t^-
                       inside the function so the model can capture
                       asymmetric reactions.
        p            : int 
                       AR lag length on y_t.
        q            : int 
                       Lag length on each side of the shock split.
        Reps         : int 
                       Total Gibbs iterations.
        burn         : int
                       Burn-in iterations.
        Nirf         : int
                       IRF horizon (in periods)
        prior_t      : int
                       Prior tightness placeholder (unused with flat prior).
        own_lag_mean : float, default 0.0
                       Prior mean on the own AR coefficients (forwarded
                       to `flat_prior`; ignored when the prior is flat).

    OUTPUT
        med_pos : numpy.ndarray of length Nirf
                  Posterior median IRF following a positive shock.
        up_pos  : numpy.ndarray of length Nirf
                  84th-percentile IRF following a positive shock.
        lo_pos  : numpy.ndarray of length Nirf
                  16th-percentile IRF following a positive shock.
        med_neg : numpy.ndarray of length Nirf
                  Posterior median IRF following a negative shock.
        up_neg  : numpy.ndarray of length Nirf
                  84th-percentile IRF following a negative shock.
        lo_neg  : numpy.ndarray of length Nirf
                  16th-percentile IRF following a negative shock.
    ---------------------------------------------------------------------
    """
    # Build the design matrix X with the +/-shock decomposition
    nx = 2
    zp = np.maximum(dz, 0)            # positive part   z_t^+
    zm = np.minimum(dz, 0)            # negative part   z_t^- (kept negative)
    dx = np.column_stack([zp, -zm])   # both parts as positive numbers

    m  = max(p, q)
    Tr = len(dy)
    T  = Tr - m                       # effective sample size

    # X = [1, dx_t, dx_{t-1}, ..., dx_{t-q}, y_{t-1}, ..., y_{t-p}].
    X = np.ones((T, 1))
    for i in range(q + 1):
        X = np.column_stack([X, dx[m - i:Tr - i, :]])
    for i in range(1, p + 1):
        X = np.column_stack([X, dy[m - i:Tr - i].reshape(-1, 1)])
    y  = dy[m:Tr]
    nc = X.shape[1]

    # OLS quantities (used as starting values and to seed sigma^2)
    XtX     = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xty     = X.T @ y
    bo      = XtX_inv @ Xty                  # OLS beta-hat
    e0      = y - X @ bo
    s2_ols  = np.dot(e0, e0) / T             # OLS variance estimate

    # Flat prior (Hm, bm) and pre-multiplied Hm * bm ──
    Hm, bm = flat_prior(y, X, nx, q, p, prior_t, own_lag_mean=own_lag_mean)
    Hm_bm  = Hm @ bm

    # Storage for the posterior IRF draws (positive / negative shock)
    Ns = Nirf + p
    Op = np.zeros((Nirf, Reps - burn))    # response to positive shock
    Om = np.zeros((Nirf, Reps - burn))    # response to negative shock

    # Initialise the chain at OLS values ──
    beta = bo.copy()
    sig2 = s2_ols

    # Gibbs iterations ──
    for jj in range(Reps):

        # sigma^2 | beta, y ~ Inverse-Gamma(T/2, SSR/2)
        # This is the univariate specialization of the Inverse-Wishart
        # IW(SSR, T) and is equivalent to Scale-Inv-Chi^2(T, SSR/T).
        eps  = y - X @ beta
        SSR  = float(np.dot(eps, eps))
        sig2 = invgamma.rvs(a=T / 2.0, scale=SSR / 2.0)

        # beta | sigma^2, y ~ Normal with the posterior covariance
        Vb  = np.linalg.inv(Hm + XtX / sig2)
        mub = Vb @ (Hm_bm + Xty / sig2)
        L   = np.linalg.cholesky(Vb)

        # Draw beta and accept only stable proposals (max 500 tries).
        ok = False
        att = 0
        while (not ok) and (att < 500):
            att += 1
            beta_new = mub + L @ np.random.randn(nc)
            ok = check_stab(beta_new[nx * (q + 1) + 1:], p)

        if ok:
            beta = beta_new

        # After burn-in, simulate the IRFs implied by this draw.
        if jj >= burn:
            ix  = jj - burn
            yp  = np.zeros(Ns)             # response to a +1 shock
            ym_ = np.zeros(Ns)             # response to a -1 shock
            for i in range(p, Ns):
                ap = am = 0.0
                # AR part (common to both responses).
                for j in range(1, p + 1):
                    ap  += yp[i - j]  * beta[nx * (q + 1) + j]
                    am  += ym_[i - j] * beta[nx * (q + 1) + j]
                # Shock-decomposition part lasts q+1 periods only.
                per = i - p
                if per <= q:
                    yp[i]  = ap + beta[1 + per * nx]         # positive part
                    ym_[i] = am + beta[1 + per * nx + 1]     # negative part
                else:
                    yp[i]  = ap
                    ym_[i] = am
            Op[:, ix] = yp[p:]
            Om[:, ix] = ym_[p:]

    # Posterior summaries: median and 16/84th percentiles ──
    return (
        np.median    (Op,         axis=1),
        np.percentile(Op, 84,     axis=1),
        np.percentile(Op, 16,     axis=1),
        np.median    (Om,         axis=1),
        np.percentile(Om, 84,     axis=1),
        np.percentile(Om, 16,     axis=1),
    )