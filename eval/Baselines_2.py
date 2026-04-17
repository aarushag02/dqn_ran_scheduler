import numpy as np


def round_robin(n_ues: int, total_prbs: int) -> list:
    """
    Allocate PRBs equally across all UEs regardless of channel quality.
    Parameters
    n_ues      : int  number of UEs
    total_prbs : int  total PRB budget
    Returns
    list of length n_ues, each value = total_prbs / n_ues
    """
    return [total_prbs / n_ues] * n_ues


def proportional_fair(state: np.ndarray, n_ues: int,
                      total_prbs: int) -> np.ndarray:
    """
    Allocate PRBs proportional to each UE's CQI value.
    The first n_ues elements of state are the CQI values.
    CQI weights are normalised to sum to 1, then multiplied by
    total_prbs to produce the allocation vector.

    Parameters
    state      : np.ndarray  shape (2*n_ues,)  [cqi_0..n, prb_0..n]
    n_ues      : int
    total_prbs : int

    Returns
    np.ndarray  shape (n_ues,)  PRB allocation per UE
    """
    cqi = np.array(state[:n_ues], dtype=np.float64)
    cqi = np.clip(cqi, 1e-9, None)          # avoid division by zero
    weights = cqi / cqi.sum()
    return (weights * total_prbs).astype(np.float32)

def max_weight(state: np.ndarray, n_ues: int, total_prbs: int) -> np.ndarray:
    """
    Allocate all available PRBs to the single UE with the highest CQI.
    This maximizes immediate throughput but completely starves other UEs.
    
    Parameters
    state      : np.ndarray  shape (2*n_ues,)  [cqi_0..n, prb_0..n]
    n_ues      : int
    total_prbs : int
    
    Returns
    np.ndarray  shape (n_ues,)  PRB allocation per UE
    """
    cqi = np.array(state[:n_ues], dtype=np.float64)
    allocation = np.zeros(n_ues, dtype=np.float32)
    
    # Find the index of the UE with the strongest channel
    best_ue_idx = np.argmax(cqi)
    
    # Assign 100% of the requested action budget to the best UE
    allocation[best_ue_idx] = float(total_prbs)
    
    return allocation