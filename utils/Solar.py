import numpy as np
from datetime import datetime as dt
from typing import List, Tuple

def hra(ts: List[dt], longitude_degrees: float):
    return np.array([15 * np.pi / 180 * ((t.hour + longitude_degrees / 15) - 12 + t.minute / 60) for t in
                     ts])  # hour angle in radians

def elevation(ts: List[dt], latitude_degrees: float, longitude_degrees: float) -> np.ndarray:
    dc = declination(ts)
    h = hra(ts, longitude_degrees)
    latitude_radians = latitude_degrees * np.pi / 180  # change degrees to radians
    # compute hour angle - the angle position of the sun for a given hour
    # compute equation: arcsin[sin(d)sin(phi)+cos(d)cos(phi)cos(hra)]
    return np.array([
        np.arcsin(np.sin(latitude_radians) * np.sin(d) +
                  np.cos(latitude_radians) * np.cos(d) * np.cos(h)) for d, h in zip(dc, h)])

def declination(ts: List[dt]) -> np.ndarray:
    ret = np.array([(t - t.replace(month=1, day=1)).days + 1 for t in ts])  # compute days since 1st january
    return np.array(
        [-23.45 * np.cos((2 * np.pi * (d + 10) / 365.25)) * np.pi / 180 for d in ret])  # compute angle in radians
