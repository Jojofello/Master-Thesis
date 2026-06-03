import pandas as pd 
import statsmodels.api as sm

def prepare_data(df, x_col, news_col, max_p, max_q):
    """
    ---------------------------------------------------------------------
    FUNCTION: prepare_data
    ---------------------------------------------------------------------
    Build a dataframe that contains the dependent variable, the news
    shock and all lagged regressors needed for the largest (p, q)
    combination of the grid. Smaller combinations later select a subset
    of these columns.

    INPUTS
        df       : pandas.DataFrame
                   Source dataframe holding x_col and news_col.
        x_col    : str
                   Column name of the dependent (return) variable.
        news_col : str
                   Column name of the structural news shock.
        max_p    : int
                   Maximum AR order to build lags for: 1, ..., max_p.
        max_q    : int
                   Maximum shock-lag order to build lags for: 0, ..., max_q.

    OUTPUT
        out : pandas.DataFrame
              Contains x_col, news_col, X_lag1..X_lag{max_p} and
              News_lag0..News_lag{max_q}, with all NaN rows dropped so
              every row is usable in the largest (p, q) model.
    ---------------------------------------------------------------------
    """
    data = df[[x_col, news_col]].copy()
    for j in range(1, max_p + 1):
        data[f"X_lag{j}"] = data[x_col].shift(j)
    for j in range(0, max_q + 1):
        data[f"News_lag{j}"] = data[news_col].shift(j)
    return data.dropna()

def run_ols(data: pd.DataFrame, x_col: str, p: int, q: int):
    """
    ---------------------------------------------------------------------
    FUNCTION: run_ols
    ---------------------------------------------------------------------
    Fit one ADL(p, q) specification by ordinary least squares.

    INPUTS
        data  : pandas.DataFrame
                Output of `prepare_data` for the same (x_col, max_p,
                max_q). The function selects only the lag columns
                actually needed for the requested (p, q).
        x_col : str
                Name of the dependent variable column.
        p     : int (>= 0)
                AR order on the dependent variable.
        q     : int (>= 0)
                Lag length on the news shock (q = 0 keeps only the
                contemporaneous News_lag0).

    OUTPUT
        aic  : float
               AIC of the fitted model (`np.inf` on failure).
        bic  : float
               BIC of the fitted model (`np.inf` on failure).
        nobs : int
               Effective number of observations used (0 on failure).
        res  : statsmodels.regression.linear_model.RegressionResults
               or None
               Full result object on success, `None` if the regression
               raised any exception.
    ---------------------------------------------------------------------
    """
    y = data[x_col]
    feature_cols = (
        [f"X_lag{j}"    for j in range(1, p + 1)] +
        [f"News_lag{j}" for j in range(0, q + 1)]
    )
    X = sm.add_constant(data[feature_cols])
    try:
        res = sm.OLS(y, X).fit()
        return res.aic, res.bic, int(res.nobs), res
    except Exception:
        return np.inf, np.inf, 0, None