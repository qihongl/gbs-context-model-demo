import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import sem, pearsonr, spearmanr
# from sklearn import metrics



def compute_stats(arr, axis=0, n_se=2, use_se=True):
    """compute mean and errorbar w.r.t to SE
    Parameters
    ----------
    arr : nd array
        data
    axis : int
        the axis to do stats along with
    n_se : int
        number of SEs
    Returns
    -------
    (n-1)d array, (n-1)d array
        mean and se
    """
    mu_ = np.mean(arr, axis=axis)
    if use_se:
        er_ = sem(arr, axis=axis) * n_se
    else:
        er_ = np.std(arr, axis=axis)
    return mu_, er_


def moving_average(x, winsize):
    return np.convolve(x, np.ones(winsize), 'valid') / winsize



def correlate_2RSMs(rsm1, rsm2, use_ranked_r=False):
    """Compute the correlation between 2 RSMs (2nd order correlations)
    Parameters
    ----------
    rsm_i: a 2d array in the form of (n_examples x n_examples)
        a representational similarity matrix

    Returns
    -------
    r: float
        linear_correlation(rsm1, rsm2)
    """
    assert np.shape(rsm1) == np.shape(rsm2)
    # only compare the lower triangular parts (w/o diagonal values)
    rsm1_vec_lower = vectorize_lower_trigular_part(rsm1)
    rsm2_vec_lower = vectorize_lower_trigular_part(rsm2)
    # compute R
    if use_ranked_r:
        r_val, p_val = spearmanr(rsm1_vec_lower, rsm2_vec_lower)
    else:
        r_val, p_val = pearsonr(rsm1_vec_lower, rsm2_vec_lower)
    return r_val, p_val


def vectorize_lower_trigular_part(matrix):
    """Exract the lower triangular entries for a matrix
        useful for computing 2nd order similarity for 2 RDMs, where
        diagonal values should be ignored
    Parameters
    ----------
    matrix: a 2d array

    Returns
    -------
    a vector of lower triangular entries
    """
    assert np.shape(matrix)[0] == np.shape(matrix)[1]
    idx_lower = np.tril_indices(np.shape(matrix)[0], -1)
    return matrix[idx_lower]
