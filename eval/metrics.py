import numpy as np


def compute_throughput(prbs: np.ndarray, cqi: np.ndarray) -> np.ndarray:
    """
    Compute per-UE throughput using the simplified Shannon formula.
        throughput_i = prbs_i * log2(1 + SNR_i)
        SNR_i        = (cqi_i - 1) * 2

    Parameters
    prbs : array-like  shape (n_ues,)  PRB allocation per UE
    cqi  : array-like  shape (n_ues,)  CQI value per UE (1–15)

    Returns
    np.ndarray  shape (n_ues,)  throughput per UE
    """
    prbs = np.asarray(prbs, dtype=np.float64)
    cqi  = np.asarray(cqi,  dtype=np.float64)
    snr  = (cqi - 1.0) * 2.0
    return prbs * np.log2(1.0 + snr)


def jains_fairness(throughputs) -> float:
    """
    Compute Jain's Fairness Index for a set of per-UE throughput values.
        J = (sum(x_i))^2 / (n * sum(x_i^2))
    A value of 1.0 means perfect fairness (all UEs receive equal throughput).
    A value of 1/n means maximum unfairness (one UE receives everything).

    Parameters
    throughputs : array-like  shape (n_ues,)

    Returns
    float in (0, 1]
    """
    t = np.asarray(throughputs, dtype=np.float64)
    sum_t  = t.sum()
    sum_t2 = (t ** 2).sum()
    if sum_t2 < 1e-12:
        return 1.0          # all zeros → technically fair (no one gets anything)
    return float(sum_t ** 2 / (len(t) * sum_t2))
