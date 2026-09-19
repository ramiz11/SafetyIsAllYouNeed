"""Observed spatial features used by the connected population procedures."""

import math


def _finite_coordinate(latitude, longitude) -> bool:
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90.0 <= latitude <= 90.0
        and -180.0 <= longitude <= 180.0
    )


def _haversine_km(left, right) -> float:
    if not _finite_coordinate(*left) or not _finite_coordinate(*right):
        return float("nan")
    lat1, lon1 = map(math.radians, map(float, left))
    lat2, lon2 = map(math.radians, map(float, right))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(
        dlon / 2.0
    ) ** 2
    return 6371.0088 * 2.0 * math.asin(min(1.0, math.sqrt(a)))



def _timestamps(frame):
    for column in ("event_time_utc", "local_time", "datetime", "timestamp"):
        if column in frame:
            return list(frame[column])
    raise ValueError("Missing observed timestamp column")
