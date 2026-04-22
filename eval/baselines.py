import numpy as np


def round_robin(n_ues: int, total_prbs: int) -> list:
    """
    Allocate PRBs equally across all UEs regardless of channel quality.

    Parameters
    ----------
    n_ues      : int  number of UEs
    total_prbs : int  total PRB budget

    Returns
    -------
    list of length n_ues, each value = total_prbs / n_ues
    """
    return [total_prbs / n_ues] * n_ues


def proportional_fair(state: np.ndarray, n_ues: int,
                      total_prbs: int) -> np.ndarray:
    """
    Allocate PRBs proportional to each UE's CQI value.

    The first n_ues elements of state are normalised CQI values in [0, 1]
    (= (raw_cqi - 1) / 14).  We recover raw CQI before computing weights
    so that the allocation is proportional to actual channel quality.
    The state is sorted by descending CQI rank; the returned allocation
    vector is in the same rank order and will be unsorting-applied by
    the environment's step().

    Parameters
    ----------
    state      : np.ndarray  shape (2*n_ues,)  normalised, CQI-rank-sorted
    n_ues      : int
    total_prbs : int

    Returns
    -------
    np.ndarray  shape (n_ues,)  PRB allocation per UE (in rank order)
    """
    cqi_norm = np.array(state[:n_ues], dtype=np.float64)
    cqi_raw  = cqi_norm * 14.0 + 1.0          # recover raw CQI in [1, 15]
    cqi_raw  = np.clip(cqi_raw, 1e-9, None)
    weights  = cqi_raw / cqi_raw.sum()
    return (weights * total_prbs).astype(np.float32)
