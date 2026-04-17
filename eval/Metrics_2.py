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

def compute_latency(throughputs: np.ndarray, packet_size_bits: float = 8192.0) -> np.ndarray:
    """
        latency_i = packet_size / throughput_i
        
    This represents the time required to clear a standard buffer/packet 
    given the UE's current data rate.
    
    Parameters
    throughputs      : array-like  shape (n_ues,)
    packet_size_bits : float       assumed buffer size to clear (default 8192 bits)
    
    Returns
    np.ndarray  shape (n_ues,)  latency per UE
    """
    t = np.asarray(throughputs, dtype=np.float64)
    # Clip at a very small number to avoid division by zero for starved UEs
    t = np.clip(t, 1e-9, None) 
    return packet_size_bits / t


def compute_energy_efficiency(total_throughput: float, active_prbs: float, 
                              p_static: float = 20.0, p_dynamic: float = 0.5) -> float:
    """
    Compute the Energy Efficiency (EE) of the cell tower in bits per Joule.
        Power_total = P_static + (P_dynamic * active_prbs)
        EE = Total_Throughput / Power_total
        
    Parameters
    total_throughput : float  Sum of all UE throughputs in the cell
    active_prbs      : float  Total number of PRBs allocated in this step
    p_static         : float  Base power consumed by the gNB (Watts)
    p_dynamic        : float  Additional power consumed per active PRB (Watts)
    
    Returns
    float  Energy Efficiency score
    """
    total_power = p_static + (active_prbs * p_dynamic)
    if total_power < 1e-9:
        return 0.0
    return float(total_throughput / total_power)