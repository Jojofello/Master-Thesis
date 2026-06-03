import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def stars(p):
    """
    ---------------------------------------------------------------------
    FUNCTION: stars
    ---------------------------------------------------------------------
    Map a p-value to the conventional significance asterisks used in the
    descriptive table.

    INPUTS
        p : float in [0, 1]
            Two-sided p-value of a hypothesis test.

    OUTPUT
        s : str
            Empty string for p >= 0.10, '*' for p < 0.10, '**' for
            p < 0.05 and '***' for p < 0.01.
    ---------------------------------------------------------------------
    """
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    else:
        return ""
    
def calculate_pvalues(df):
    """
    ---------------------------------------------------------------------
    FUNCTION: calculate_pvalues
    ---------------------------------------------------------------------
    Calculate Pearson correlation coefficients (r) and the associated 
    significance levels (p-values) for all columns in a DataFrame.

    INPUTS
        df : pd.DataFrame
            A pandas DataFrame containing numeric time-series data.

    OUTPUTS
        r_matrix : pd.DataFrame
            Symmetric matrix of Pearson correlation coefficients.
        p_matrix : pd.DataFrame
            Symmetric matrix of the corresponding p-values.
    ---------------------------------------------------------------------
    """
    cols = df.columns
    n = len(cols)
    
    r_matrix = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)
    p_matrix = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
    
    for i in range(n):
        for j in range(i + 1, n):
            r, p = pearsonr(df.iloc[:, i], df.iloc[:, j])
            r_matrix.iloc[i, j] = r_matrix.iloc[j, i] = r
            p_matrix.iloc[i, j] = p_matrix.iloc[j, i] = p
            
    return r_matrix, p_matrix