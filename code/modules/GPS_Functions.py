import math
import numpy


def get_relative_coordinates(lat_ref, long_ref, lat_meas, long_meas):
    # Outputs the relative East-North coordinates in meters
    # The inputs are the reference position and measurement postion in [deg]
    # We could make something smart to avoid recalculating the curvation radius of the given latitude

    degrees_to_radians = math.pi / 180.0
    R_M = 6378137  # [m]  %semi-major axis
    e_earth = 0.0818191908426  # Earth Ellipticity
    R_EW = R_M / ((1 - (e_earth ** 2) * (math.sin(lat_ref) ** 2)) ** (1 / 2))
    R_NS = (R_M * (1 - e_earth ** 2)) / ((1 - (e_earth ** 2) * (math.sin(lat_ref) ** 2)) ** (3 / 2))
    cos_lat_ref = math.cos(lat_ref * degrees_to_radians)

    east_west_distance = cos_lat_ref * R_EW * (long_meas - long_ref) * degrees_to_radians
    north_south_distance = R_NS * (lat_meas - lat_ref) * degrees_to_radians

    return east_west_distance, north_south_distance


def get_distance(lat_ref, long_ref, lat_meas, long_meas):
    east_west_distance, north_south_distance = get_relative_coordinates(lat_ref, long_ref, lat_meas, long_meas)
    distance = (east_west_distance ** 2 + north_south_distance ** 2) ** 0.5

    return distance
