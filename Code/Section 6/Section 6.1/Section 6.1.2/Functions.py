import numpy as np 
from arch import arch_model
from scipy.special import gammaln

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

def egarch_t_E_abs_z(nu):
    """
    ---------------------------------------------------------------------
    FUNCTION: egarch_t_E_abs_z
    ---------------------------------------------------------------------
    Return E[|z|] when z follows a *standardised* (variance-one) Student-t
    distribution with `nu` degrees of freedom. For the Gaussian case
    this constant equals sqrt(2/pi); for the t-distribution it is the
    distribution-specific centring term used in Nelson's (1991) EGARCH
    recursion.

    INPUTS
        nu : float (> 2)
             Degrees of freedom of the standardised Student-t.

    OUTPUT
        c : float
            E[|z|] for z ~ standardised t(nu).
    ---------------------------------------------------------------------
    """
    return (2.0 * np.sqrt(nu - 2.0)
            * np.exp(gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0))
            / ((nu - 1.0) * np.sqrt(np.pi)))

def garch_sigma2(eps, theta_vol, mtype, c=None):
    """
    ---------------------------------------------------------------------
    FUNCTION: garch_sigma2
    ---------------------------------------------------------------------
    Conditional-variance recursion for one of the supported GARCH
    families. The recursion is started at the unconditional sample
    variance of `eps` (clipped from below to keep the start non-zero).

    Variance equations:
        GARCH(1,1)
            sigma2_t = omega + alpha eps_{t-1}^2 + beta sigma2_{t-1}.
        GJR(1,1,1)  with negative-shock indicator I[eps<0]
            sigma2_t = omega + alpha eps_{t-1}^2
                       + gamma I[eps_{t-1}<0] eps_{t-1}^2
                       + beta sigma2_{t-1}.
        EGARCH(1,1,1)
            log sigma2_t = omega + beta log sigma2_{t-1}
                           + alpha (|z_{t-1}| - c) + gamma z_{t-1},
            with z_{t-1} = eps_{t-1} / sigma_{t-1}.

    INPUTS
        eps       : 1-D numpy.ndarray of length T
                    Residual series y - X beta of the mean equation.
        theta_vol : 1-D numpy.ndarray
                    GARCH parameter vector. Length 3 for GARCH (omega,
                    alpha, beta), length 4 for GJR/EGARCH (omega, alpha,
                    gamma, beta).
        mtype     : str in {'GARCH', 'GJR', 'EGARCH'}
                    Variance-equation specification.
        c         : float or None, default None
                    EGARCH centring constant E[|z|] under the assumed
                    innovation distribution. Use sqrt(2/pi) for Gaussian
                    z; use `egarch_t_E_abs_z(nu)` for standardised
                    Student-t. Ignored unless mtype == 'EGARCH'. If None
                    and mtype == 'EGARCH', sqrt(2/pi) is used (the same
                    convention as the `arch` package).

    OUTPUT
        s2 : numpy.ndarray of length T
             Conditional variance series sigma2_t.
    ---------------------------------------------------------------------
    """
    T  = len(eps)
    s2 = np.empty(T)
    var0 = max(np.var(eps), 1e-12)

    if mtype == 'GARCH':
        omega, alpha, beta = theta_vol
        s2[0] = var0
        for t in range(1, T):
            s2[t] = omega + alpha * eps[t-1]**2 + beta * s2[t-1]

    elif mtype == 'GJR':
        omega, alpha, gamma, beta = theta_vol
        s2[0] = var0
        for t in range(1, T):
            neg = 1.0 if eps[t-1] < 0 else 0.0
            s2[t] = (omega
                     + alpha * eps[t-1]**2
                     + gamma * neg * eps[t-1]**2
                     + beta  * s2[t-1])

    elif mtype == 'EGARCH':
        omega, alpha, gamma, beta = theta_vol
        ls = np.empty(T)
        ls[0] = np.log(var0)
        if c is None:
            c = np.sqrt(2.0 / np.pi)
        for t in range(1, T):
            sig_prev = np.sqrt(np.exp(np.clip(ls[t-1], -30, 30)))
            z = eps[t-1] / sig_prev
            ls[t] = omega + beta * ls[t-1] + alpha * (abs(z) - c) + gamma * z
        s2 = np.exp(np.clip(ls, -30, 30))

    return s2

def _split_theta(theta, dist):
    """
    ---------------------------------------------------------------------
    FUNCTION: _split_theta
    ---------------------------------------------------------------------
    Split the joint variance-parameter vector into a GARCH part and a
    degrees-of-freedom scalar.

    INPUTS
        theta : 1-D array-like
                Either the bare GARCH parameters (Gaussian innovations)
                or the GARCH parameters with `nu` appended (Student-t
                innovations).
        dist  : str in {'normal', 't'}
                Innovation distribution flag. When 't', the last entry
                of `theta` is the Student-t degrees of freedom.

    OUTPUT
        theta_vol : numpy.ndarray
                    GARCH-equation parameters only (no `nu`).
        nu        : float or None
                    Student-t degrees of freedom, or None when
                    dist == 'normal'.
    ---------------------------------------------------------------------
    """
    if dist == 't':
        return np.asarray(theta[:-1]), float(theta[-1])
    return np.asarray(theta), None

def garch_valid(theta, mtype, dist='normal'):
    """
    ---------------------------------------------------------------------
    FUNCTION: garch_valid
    ---------------------------------------------------------------------
    Indicator function describing the support of the GARCH-family
    likelihood. Used both to reject invalid MH proposals and to short-
    circuit the log-likelihood evaluation.

    Validity conditions (in addition to nu > 2.01 when dist == 't'):
        GARCH    omega > 0,  alpha >= 0,  beta >= 0,  alpha + beta < 1.
        GJR      omega > 0,  alpha >= 0,  alpha + gamma >= 0,  beta >= 0,
                 alpha + 0.5 gamma + beta < 1.
        EGARCH   |alpha|, |gamma| < 5,  |beta| < 1.

    INPUTS
        theta : 1-D array-like
                Joint variance-parameter vector (with `nu` appended when
                dist == 't').
        mtype : str in {'GARCH', 'GJR', 'EGARCH'}
                Variance-equation specification.
        dist  : str in {'normal', 't'}, default 'normal'
                Innovation distribution.

    OUTPUT
        ok : bool
             True if `theta` lies inside the support, False otherwise.
    ---------------------------------------------------------------------
    """
    theta_vol, nu = _split_theta(theta, dist)
    if dist == 't' and not (nu > 2.01 and nu < 300.0):
        return False
    if mtype == 'GARCH':
        if len(theta_vol) != 3: return False
        o, a, b = theta_vol
        return (o > 1e-14) and (a >= 0) and (b >= 0) and (a + b < 0.9995)
    if mtype == 'GJR':
        if len(theta_vol) != 4: return False
        o, a, g, b = theta_vol
        return ((o > 1e-14) and (a >= 0) and (a + g >= 0) and (b >= 0)
                and (a + 0.5 * g + b < 0.9995))
    if mtype == 'EGARCH':
        if len(theta_vol) != 4: return False
        o, a, g, b = theta_vol
        return (abs(b) < 0.9995) and (abs(a) < 5) and (abs(g) < 5)
    return False

def garch_loglik(eps, theta, mtype, dist='normal'):
    """
    ---------------------------------------------------------------------
    FUNCTION: garch_loglik
    ---------------------------------------------------------------------
    Log-likelihood of `eps` under the chosen GARCH-family variance
    process and innovation distribution. Returns -inf whenever the
    parameters fall outside the support or the implied variance series
    is non-positive / non-finite.

    Densities used:
        Gaussian      0.5 * sum( log(2 pi s2_t) + eps_t^2 / s2_t ).
        Student-t     Standardised t(nu) density evaluated at the
                      standardised innovation eps_t / sqrt(s2_t), with
                      the variance-rescaling factor s2_t handled
                      explicitly so that Var[eps_t | F_{t-1}] = s2_t.

    INPUTS
        eps   : 1-D numpy.ndarray of length T
                Residual series of the mean equation.
        theta : 1-D array-like
                Joint variance-parameter vector (with `nu` appended when
                dist == 't').
        mtype : str in {'GARCH', 'GJR', 'EGARCH'}
        dist  : str in {'normal', 't'}, default 'normal'.

    OUTPUT
        ll : float
             Log-likelihood; -inf if parameters are invalid or numerical
             issues arise.
    ---------------------------------------------------------------------
    """
    if not garch_valid(theta, mtype, dist):
        return -np.inf
    theta_vol, nu = _split_theta(theta, dist)

    # Distribution-aware EGARCH centring (Nelson 1991): c = E[|z|] under
    # the assumed innovation distribution. Unused for non-EGARCH models.
    if mtype == 'EGARCH':
        c = egarch_t_E_abs_z(nu) if dist == 't' else np.sqrt(2.0 / np.pi)
    else:
        c = None

    s2 = garch_sigma2(eps, theta_vol, mtype, c=c)
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return -np.inf

    if dist == 'normal':
        ll = -0.5 * np.sum(np.log(2 * np.pi * s2) + eps**2 / s2)
    else:
        T_    = len(eps)
        const = (gammaln((nu + 1) / 2.0)
                 - gammaln(nu / 2.0)
                 - 0.5 * np.log(np.pi * (nu - 2.0)))
        ll = (T_ * const
              - 0.5 * np.sum(np.log(s2))
              - 0.5 * (nu + 1.0) * np.sum(
                  np.log1p(eps**2 / ((nu - 2.0) * s2))))
    return ll if np.isfinite(ll) else -np.inf

def select_variance_model(eps, verbose=False):
    """
    ---------------------------------------------------------------------
    FUNCTION: select_variance_model
    ---------------------------------------------------------------------
    Pick the {variance model x innovation distribution} combination that
    minimises BIC among
        {GARCH, GJR, EGARCH} x {Gaussian, standardised Student-t}.

    The series `eps` is rescaled to unit standard deviation before the
    `arch` package is called so the optimiser sees a well-conditioned
    problem; the resulting parameters are then unscaled back to the
    original scale of `eps` so they can be used directly in
    `garch_sigma2`.

    Note on EGARCH: the `arch` package always uses c = sqrt(2/pi) in
    its EGARCH recursion, regardless of the assumed innovation
    distribution. To allow downstream code to use the distribution-
    correct centring c_t(nu) of Nelson (1991) under Student-t innovations,
    `omega` is bijectively shifted by alpha * (c_t(nu) - sqrt(2/pi)).
    The conditional variances are numerically identical; only the
    meaning of `omega` changes.

    INPUTS
        eps     : 1-D numpy.ndarray of length T
                  Residual series on which to fit the variance models.
        verbose : bool, default False
                  If True, print one line per (model, distribution)
                  combination and flag the selected one.

    OUTPUT
        best_name : str
                    Selected variance family ('GARCH', 'GJR' or 'EGARCH').
        best_dist : str
                    Selected innovation distribution ('normal' or 't').
        theta     : numpy.ndarray
                    Variance-equation parameters in original-scale form,
                    with `nu` appended when best_dist == 't'.
        table     : list of (name, dist, BIC, AIC, logL)
                    Diagnostic table of every fitted candidate; failed
                    fits enter as (np.inf, np.inf, -np.inf).
    ---------------------------------------------------------------------
    """
    sd    = np.std(eps)
    scale = 1.0 / sd if sd > 0 else 1.0
    eps_s = eps * scale

    specs = [
        ('GARCH',  'normal', dict(vol='GARCH',  p=1, q=1)),
        ('GJR',    'normal', dict(vol='GARCH',  p=1, o=1, q=1)),
        ('EGARCH', 'normal', dict(vol='EGARCH', p=1, o=1, q=1)),
        ('GARCH',  't',      dict(vol='GARCH',  p=1, q=1)),
        ('GJR',    't',      dict(vol='GARCH',  p=1, o=1, q=1)),
        ('EGARCH', 't',      dict(vol='EGARCH', p=1, o=1, q=1)),
    ]
    table     = []
    best_name = None
    best_dist = None
    best_res  = None
    best_bic  = np.inf

    #   Fit every candidate; track the BIC-minimiser.
    for name, dist, spec in specs:
        try:
            am = arch_model(eps_s, mean='Zero', dist=dist, **spec)
            r  = am.fit(disp='off', show_warning=False)
            table.append((name, dist, r.bic, r.aic, r.loglikelihood))
            if r.bic < best_bic:
                best_bic  = r.bic
                best_name = name
                best_dist = dist
                best_res  = r
        except Exception:
            table.append((name, dist, np.inf, np.inf, -np.inf))

    #   Unscale the chosen parameters back to the original scale of eps.
    p = best_res.params.values
    if best_name == 'GARCH':
        theta_vol = np.array([p[0] / scale**2, p[1], p[2]])
    elif best_name == 'GJR':
        theta_vol = np.array([p[0] / scale**2, p[1], p[2], p[3]])
    else:  # EGARCH
        alpha   = p[1]
        gamma   = p[2]
        beta    = p[3]
        omega_u = p[0] - 2 * np.log(scale) * (1.0 - beta)
        if best_dist == 't':
            nu_hat   = float(p[-1])
            c_t      = egarch_t_E_abs_z(nu_hat)
            omega_u += alpha * (c_t - np.sqrt(2.0 / np.pi))
        theta_vol = np.array([omega_u, alpha, gamma, beta])

    #   Append `nu` if Student-t innovations were chosen.
    if best_dist == 't':
        nu    = float(p[-1])
        theta = np.concatenate([theta_vol, [nu]])
    else:
        theta = theta_vol

    if verbose:
        print("   Variance-model + distribution selection (BIC):")
        for n, d, b, a, l in table:
            d_abbr = 'N' if d == 'normal' else 't'
            flag   = "  ← selected" if (n == best_name and d == best_dist) else ""
            print(f"     {n:6s}-{d_abbr}  BIC={b:10.3f}  "
                  f"AIC={a:10.3f}  logL={l:10.3f}{flag}")
    return best_name, best_dist, theta, table

def run_irf_single_mh(dy, dz, p, q, Reps, burn, Nirf, prior_t, tune_window, target_acc, verbose=False):
    """
    ---------------------------------------------------------------------
    FUNCTION: run_irf_single_mh
    ---------------------------------------------------------------------
    Run the single-block Metropolis-Hastings sampler for the ADL model
    with a GARCH-family residual variance and produce posterior IRF
    summaries (median + 16/84th percentiles) for both a positive and a
    negative one-unit shock.

    The sampler mirrors the design-matrix construction of the
    constant-variance Gibbs version so the two are directly comparable.

    INPUTS
        dy      : 1-D numpy.ndarray of length Tr
                  Dependent variable y_t.
        dz      : 1-D numpy.ndarray of length Tr
                  Shock series z_t (split into +/- parts inside).
        p       : int (non-negative)   AR lag length on y_t.
        q       : int (non-negative)   Lag length on each side of z_t.
        Reps         : int 
                       Total Gibbs iterations.
        burn         : int
                       Burn-in iterations.
        Nirf         : int
                       IRF horizon (in periods)
        prior_t      : int
                       Prior tightness placeholder (unused with flat prior).
        tune_window  : int
                       Adapt scales for the first 'tune_window' iters.
        target_acc   : int
                       Target acceptance rate during burn-in. 
        verbose : bool, default False
                  Forward to `select_variance_model`.

    OUTPUT
        out : dict with keys
              mP, uP, lP : posterior median, 84%- and 16%-percentile
                           IRFs (length Nirf) under a positive shock.
              mN, uN, lN : same triple under a negative shock.
              vmodel     : str         selected variance family.
              vdist      : str         selected innovation distribution.
              vparams    : ndarray     final theta (last MCMC state).
              vtable     : list        BIC-selection diagnostic table.
              acc_rate   : float       overall MH acceptance rate.
              c_beta_final, c_theta_final : final adapted RW scales.
    ---------------------------------------------------------------------
    """
    #    Design matrix (identical to the constant-variance version) ──
    nx = 2
    zp = np.maximum(dz, 0)
    zm = np.minimum(dz, 0)
    dx = np.column_stack([zp, -zm])
    m  = max(p, q)
    Tr = len(dy)
    T  = Tr - m
    X = np.ones((T, 1))
    for i in range(q + 1):
        X = np.column_stack([X, dx[m - i:Tr - i, :]])
    for i in range(1, p + 1):
        X = np.column_stack([X, dy[m - i:Tr - i].reshape(-1, 1)])
    y  = dy[m:Tr]
    nc = X.shape[1]

    #    OLS starting values ─────────────────────────────────────────
    XtX     = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    bo      = XtX_inv @ (X.T @ y)
    eps_ols = y - X @ bo
    s2_ols  = np.dot(eps_ols, eps_ols) / T

    #    Variance model + distribution selection on the OLS residuals
    vtype, vdist, theta_hat, vtable = select_variance_model(eps_ols, verbose=verbose)
    n_theta = len(theta_hat)        # includes nu when vdist == 't'

    #    Flat prior (kept diagonal for speed) ────────────────────────
    Hm, bm  = flat_prior(y, X, nx, q, p, prior_t)
    Hm_diag = np.diag(Hm)

    #    Initial state of the Markov chain ───────────────────────────
    beta  = bo.copy()
    theta = theta_hat.copy()

    #   Per-coordinate proposal scales.
    #       sd_beta uses the OLS sandwich covariance as a Laplace proxy;
    #       sd_theta is a 5 % perturbation of the |theta_hat| magnitudes.
    sd_beta  = np.sqrt(np.maximum(s2_ols * np.diag(XtX_inv), 1e-14))
    sd_theta = 0.05 * np.abs(theta_hat) + 1e-4
    c_beta   = 2.38 / np.sqrt(nc)
    c_theta  = 2.38 / np.sqrt(n_theta)

    #   Joint log-posterior ─────────────────────────────────────────
    def log_posterior(beta_, theta_):
        # 1. VAR-stability indicator prior on beta.
        if not check_stab(beta_[nx * (q + 1) + 1:], p):
            return -np.inf
        # 2. GARCH-support indicator prior on theta (and nu > 2 if t).
        eps_ = y - X @ beta_
        ll   = garch_loglik(eps_, theta_, vtype, vdist)
        if not np.isfinite(ll):
            return -np.inf
        return ll

    lpost_cur = log_posterior(beta, theta)
    if not np.isfinite(lpost_cur):
        # Restart at the convex combination of the OLS and prior means
        # in case the OLS coefficients leak outside the stability set.
        beta = 0.5 * bo + 0.5 * bm
        lpost_cur = log_posterior(beta, theta)

    #   Storage and bookkeeping ─────────────────────────────────────
    Op = np.zeros((Nirf, Reps - burn))
    Om = np.zeros((Nirf, Reps - burn))
    Ns = Nirf + p
    n_acc_total = 0
    n_acc_block = 0
    n_prop_block = 0

    #   Main MH loop ────────────────────────────────────────────────
    for jj in range(Reps):

        #   Joint random-walk proposal psi* = psi + epsilon.
        beta_prop  = beta  + c_beta  * sd_beta  * np.random.randn(nc)
        theta_prop = theta + c_theta * sd_theta * np.random.randn(n_theta)

        lpost_new = log_posterior(beta_prop, theta_prop)

        #   Standard MH acceptance.
        if np.log(np.random.rand()) < (lpost_new - lpost_cur):
            beta      = beta_prop
            theta     = theta_prop
            lpost_cur = lpost_new
            n_acc_block += 1
            n_acc_total += 1
        n_prop_block += 1

        #   Adapt the scales every 100 iters during burn-in.
        if (jj < tune_window) and (jj > 0) and (jj % 100 == 0):
            rate = n_acc_block / max(n_prop_block, 1)
            if rate < target_acc - 0.05:
                c_beta  *= 0.85
                c_theta *= 0.85
            elif rate > target_acc + 0.10:
                c_beta  *= 1.15
                c_theta *= 1.15
            n_acc_block  = 0
            n_prop_block = 0

        #   Store the IRF implied by this draw, after burn-in.
        if jj >= burn:
            ix  = jj - burn
            yp  = np.zeros(Ns)
            ym_ = np.zeros(Ns)
            for i in range(p, Ns):
                ap = am_ = 0.0
                # AR part (common to + and - responses).
                for j in range(1, p + 1):
                    ap  += yp[i - j]  * beta[nx * (q + 1) + j]
                    am_ += ym_[i - j] * beta[nx * (q + 1) + j]
                # Shock-decomposition part lasts q+1 periods only.
                per = i - p
                if per <= q:
                    yp[i]  = ap  + beta[1 + per * nx]
                    ym_[i] = am_ + beta[1 + per * nx + 1]
                else:
                    yp[i]  = ap
                    ym_[i] = am_
            Op[:, ix] = yp[p:]
            Om[:, ix] = ym_[p:]

    #   Posterior summaries packaged into a dictionary ──────────────
    return dict(
        mP = np.median    (Op,         axis=1),
        uP = np.percentile(Op, 84,     axis=1),
        lP = np.percentile(Op, 16,     axis=1),
        mN = np.median    (Om,         axis=1),
        uN = np.percentile(Om, 84,     axis=1),
        lN = np.percentile(Om, 16,     axis=1),
        vmodel        = vtype,
        vdist         = vdist,
        vparams       = theta,
        vtable        = vtable,
        acc_rate      = n_acc_total / Reps,
        c_beta_final  = c_beta,
        c_theta_final = c_theta,
    )
    
def _vm_label(vmodel, vdist, vparams):
    """
    ---------------------------------------------------------------------
    FUNCTION: _vm_label
    ---------------------------------------------------------------------
    Compose a short LaTeX-aware label describing the variance model and
    the innovation distribution that was selected for a given panel
    (e.g. 'GARCH-N' or 'GJR-t ($\\nu$ = 5.32)').

    INPUTS
        vmodel  : str
                  Variance-family name ('GARCH', 'GJR', 'EGARCH').
        vdist   : str
                  Innovation distribution ('normal' or 't').
        vparams : 1-D array-like
                  Variance-equation parameters; the last entry is `nu`
                  when vdist == 't'.

    OUTPUT
        s : str
            Formatted label suitable for a matplotlib annotation.
    ---------------------------------------------------------------------
    """
    if vdist == 't':
        nu = vparams[-1]
        return f'{vmodel}-t ($\\nu$ = {nu:.2f})'
    return f'{vmodel}-N'