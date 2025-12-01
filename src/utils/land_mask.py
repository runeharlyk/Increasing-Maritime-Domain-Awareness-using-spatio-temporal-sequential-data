import numpy as np
import torch
from pathlib import Path

try:
    from global_land_mask import globe

    HAS_GLOBE = True
except ImportError:
    HAS_GLOBE = False
    print("Warning: global_land_mask not installed. Install with: pip install global-land-mask")


class LandMask:

    def __init__(self, bounds=None):
        if bounds is None:
            bounds = [60, 0, 50, 20]

        self.north, self.west, self.south, self.east = bounds

        if not HAS_GLOBE:
            raise ImportError("global_land_mask library required. Install with: pip install global-land-mask")

    def in_bbox(self, lat, lon):
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        return (lat >= self.south) & (lat <= self.north) & (lon >= self.west) & (lon <= self.east)

    def is_water_or_land(self, lat, lon):
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        inside = self.in_bbox(lat, lon)
        land = globe.is_land(lat, lon)
        water = ~land
        land = land & inside
        water = water & inside
        return land, water, inside

    def check_land(self, lat, lon):
        land, water, inside = self.is_water_or_land(lat, lon)
        return land.astype(np.float32)

    def check_land_torch(self, lat, lon):
        lat_np = lat.detach().cpu().numpy()
        lon_np = lon.detach().cpu().numpy()
        land_np = self.check_land(lat_np, lon_np)
        return torch.from_numpy(land_np).to(lat.device)

    def get_land_penalty(self, lat, lon):
        return self.check_land_torch(lat, lon)
