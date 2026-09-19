from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable


US_SURVEY_FOOT_METERS = 1200.0 / 3937.0


@dataclass(frozen=True)
class SafetySemantics:
    name: str
    projected_crs: int
    nominal_buffer: float
    crime_window_weeks: int
    aggregation: str
    route_origin: str
    omit_invalid_predictions: bool
    nyc_major_offenses_only: bool

    @property
    def effective_buffer_meters(self) -> float:
        if self.projected_crs == 2263:
            return self.nominal_buffer * US_SURVEY_FOOT_METERS
        return self.nominal_buffer


def robust_scale(value: float, stats: dict[str, float]) -> float:
    iqr = float(stats["75%"]) - float(stats["25%"])
    if iqr == 0:
        return 0.0
    return min(1.0, max(0.0, (float(value) - float(stats["50%"])) / iqr))


def normalized_safety(crime_count: float, stats: dict[str, float]) -> float:
    return 1.0 - robust_scale(crime_count, stats)


def aggregate_safety(
    scores: Iterable[float | None],
    *,
    total_predictions: int,
    aggregation: str,
) -> dict:
    valid = [float(value) for value in scores if value is not None]
    # ``statistics.fmean`` was only added in Python 3.8.  The publication
    # repository still declares/supports a Python 3.7-era environment, and
    # these inputs are already normalized to ``float`` above.
    mean = sum(valid) / len(valid) if valid else None
    median = statistics.median(valid) if valid else None
    if aggregation == "median":
        value = median
    elif aggregation == "mean":
        value = mean
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    return {
        "value": value,
        "aggregation": aggregation,
        "mean": mean,
        "median": median,
        "valid_routes": len(valid),
        "total_predictions": total_predictions,
        "coverage": len(valid) / total_predictions if total_predictions else 0.0,
    }
