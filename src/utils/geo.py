import numpy as np
import torch


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def haversine_distance_torch(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1_rad = torch.deg2rad(lat1)
    lat2_rad = torch.deg2rad(lat2)
    lon1_rad = torch.deg2rad(lon1)
    lon2_rad = torch.deg2rad(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
    a = torch.clamp(a, 0.0, 1.0) 
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a + 1e-7))

    distance = R * c
    return distance
