import numpy as np
from datetime import datetime as dt
from datetime import date
from typing import List, Tuple

def lst_timestamp(ts: List[int], longitude_degrees: float):
    ts = [dt.fromtimestamp(t) for t in ts]
    return lst(ts, longitude_degrees)

def lst(ts: List[dt], longitude_degrees: float):
    return np.array([((t.hour + longitude_degrees / 15) + t.minute / 60) for t in ts])

def hra_timestamp(ts: List[dt], longitude_degrees: float):
    ts = [dt.fromtimestamp(t) for t in ts]
    return hra(ts, longitude_degrees)

def hra(ts: List[dt], longitude_degrees: float):
    return 15 * np.pi / 180 * (lst(ts, longitude_degrees) - 12)  # hour angle in radians

def elevation(ts: List[dt], latitude_degrees: float, longitude_degrees: float) -> np.ndarray:
    dc = declination_rad(ts)
    h = hra(ts, longitude_degrees)
    latitude_radians = latitude_degrees * np.pi / 180  # change degrees to radians
    # compute hour angle - the angle position of the sun for a given hour
    # compute equation: arcsin[sin(d)sin(phi)+cos(d)cos(phi)cos(hra)]
    return np.array([
        np.arcsin(np.sin(latitude_radians) * np.sin(d) +
                  np.cos(latitude_radians) * np.cos(d) * np.cos(h)) for d, h in zip(dc, h)])

def declination_rad(ts: List[dt]) -> np.ndarray:
    ret = np.array([(t - t.replace(month=1, day=1)).days + 1 for t in ts])  # compute days since 1st january
    return np.array(
        [-23.45 * np.cos((2 * np.pi * (d + 10) / 365.25)) * np.pi / 180 for d in ret])  # compute angle in radians

def sun_maximum_positive_elevation(latitude):
    latitude_radians = latitude * np.pi / 180
    #return 90 - 23.45 + latitude
    return np.arcsin(np.cos(latitude_radians) * np.cos(23.45*np.pi/180) + np.sin(latitude_radians) * np.sin(23.45*np.pi/180)) * 180 / np.pi

def zenith_elevation(ts: List[int], latitude_degrees):
    # print(timestamp)
    # dt_ = dt.fromtimestamp(timestamp)
    ts = [dt.fromtimestamp(t) for t in ts]
    ts = [t.replace(hour=11, minute=0, second=0) for t in ts]
    # dt.hour = 11 # range(24) = 0..23
    # dt.minutes = 0
    # dt.seconds = 0

    # ts = dt.timestamp(dt_)
    dec_radians = declination_rad(ts)
    # hra = SolarInsulation.hra(ts)
    hra = 0 # zenith is when hra == 0
    latitude_radians = latitude_degrees * np.pi / 180
    # compute hour angle - the angle position of the sun for a given hour
    # compute equation: sin[sin(d)sin(phi)+cos(d)cos(phi)cos(hra)]
    # print(np.pi/2 - dec_radians)
    #return np.cos(latitude_radians) * np.cos(dec_radians), np.sin(latitude_radians) * np.sin(dec_radians)
    return np.arcsin(np.cos(latitude_radians) * np.cos(dec_radians) + np.sin(latitude_radians) * np.sin(dec_radians)) * 180 / np.pi

    #return  np.sin(dec_radians) + np.cos(dec_radians)

def sunrise_timestamp(ts: List[int], latitude_degrees):
    ts = [dt.fromtimestamp(t) for t in ts]
    dc = declination_rad(ts)
    latitude_radians = latitude_degrees * np.pi / 180  # change degrees to radians
    return  12 - (1/15) * np.arccos(-np.tan(latitude_radians) * np.tan(dc)) * 180 / np.pi

def sunset_timestamp(ts: List[int], latitude_degrees):
    ts = [dt.fromtimestamp(t) for t in ts]
    dc = declination_rad(ts)
    latitude_radians = latitude_degrees * np.pi / 180  # change degrees to radians
    return  12 + (1/15) * np.arccos(-np.tan(latitude_radians) * np.tan(dc)) * 180 / np.pi

def longest_day(latitude_degrees):
    solistice = dt(2025, 6, 21).timestamp() #21.06.2025
    longest_day = sunset_timestamp([solistice],latitude_degrees) - sunrise_timestamp([solistice],latitude_degrees)
    return longest_day[0]