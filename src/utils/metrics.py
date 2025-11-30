
from src.utils import haversine_distance
import numpy as np

def pointwise_haversine(y_true, y_pred):
    """
    Compute haversine distance at each time step, for each item in batch.

    y_true, y_pred: shape (B, T, 2) with [..., 0]=lat_deg, [..., 1]=lon_deg
    Returns: distances with shape (B, T) in meters.
    """

    lat_true = y_true[..., 0]
    lon_true = y_true[..., 1]
    lat_pred = y_pred[..., 0]
    lon_pred = y_pred[..., 1]

    return haversine_distance(lat_true, lon_true, lat_pred, lon_pred)


## Mean Haversine Error

def mean_haversine_error(y_true, y_pred):
    """
    Mean haversine error over all batch items and time steps.
    """
    dists = pointwise_haversine(y_true, y_pred)  # (B, T)
    return float(dists.mean())


## Root Mean Squared Error (RMSE)

def rmse_haversine(y_true, y_pred):
    """
    RMSE of haversine distances in meters.
    """
    dists = pointwise_haversine(y_true, y_pred)  # (B, T)
    return float(np.sqrt(np.mean(dists ** 2))) 


## Average Displacement Error (ADE)

def ade(y_true, y_pred):
    """
    ADE: Average Displacement Error over whole trajectories.
    Here it's effectively the same as mean_haversine_error
    """
    dists = pointwise_haversine(y_true, y_pred)  # (B, T)
    return float(dists.mean())

## Final Displacement Error (FDE)

def fde(y_true, y_pred):
    """
    FDE: error at the final time step (averaged over batch).
    """
    dists = pointwise_haversine(y_true, y_pred)  # (B, T)
    final_dists = dists[:, -1]  # last time step
    return float(final_dists.mean())


## Dynamic Time Warping Distance (DTW)

def dtw_distance_trajectory(traj_true, traj_pred):
    """
    DTW distance between two single trajectories.

    traj_true, traj_pred: shape (T, 2) with [lat_deg, lon_deg]
    Returns: DTW distance in meters (normalized by path length).
    """

    T1 = traj_true.shape[0]
    T2 = traj_pred.shape[0]

    # Cost matrix: pairwise haversine distances
    # Shape (T1, T2)
    lat1 = traj_true[:, 0][:, None]  # (T1, 1)
    lon1 = traj_true[:, 1][:, None]  # (T1, 1)
    lat2 = traj_pred[:, 0][None, :]  # (1, T2)
    lon2 = traj_pred[:, 1][None, :]  # (1, T2)

    cost = haversine_distance(lat1, lon1, lat2, lon2)

    # Accumulated cost matrix
    acc = np.zeros((T1, T2), dtype=np.float64)
    acc[0, 0] = cost[0, 0]

    # First row
    for j in range(1, T2):
        acc[0, j] = cost[0, j] + acc[0, j - 1]

    # First column
    for i in range(1, T1):
        acc[i, 0] = cost[i, 0] + acc[i - 1, 0]

    # Rest
    for i in range(1, T1):
        for j in range(1, T2):
            acc[i, j] = cost[i, j] + min(
                acc[i - 1, j],      # insertion
                acc[i, j - 1],      # deletion
                acc[i - 1, j - 1],  # match
            )

    # Normalized by path length 
    path_length = T1 + T2  # rough normalization
    return float(acc[-1, -1] / path_length)


def dtw_batch_mean(y_true, y_pred):
    """
    Mean DTW distance over a batch of trajectories.

    y_true, y_pred: shape (B, T, 2)
    Returns: scalar, average DTW distance in meters.
    """

    B = y_true.shape[0]
    dtw_values = []

    for b in range(B):
        dtw_b = dtw_distance_trajectory(y_true[b], y_pred[b])
        dtw_values.append(dtw_b)

    return float(np.mean(dtw_values))
