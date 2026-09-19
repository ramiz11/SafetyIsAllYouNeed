"""Connected population procedures calibrated from training and observed histories."""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def _history(frame):
    history = frame.iloc[:-1]
    if history.empty:
        raise ValueError("A trajectory must contain observed history and a target")
    return history


def _times(history):
    for column in ("local_time", "event_time_utc", "utc_time"):
        if column in history:
            values = [pd.Timestamp(value) for value in history[column]]
            if any(pd.isna(value) for value in values):
                raise ValueError("Missing observed timestamp")
            return values
    raise ValueError("Missing observed timestamp column")


def observed_session_counts(frame, gap_minutes):
    """Count events and sessions using only the prefix, in its stored order.

    A new session starts after a gap strictly greater than the threshold or a
    backwards time step. Zero gaps remain in the same session. The threshold is
    supplied by the caller and must not be estimated from the held-out row.
    """
    if not np.isfinite(gap_minutes) or gap_minutes < 0:
        raise ValueError("Session gap must be finite and nonnegative")
    times = _times(_history(frame))
    gaps = [(right - left).total_seconds() / 60 for left, right in zip(times, times[1:])]
    connected = [0 <= gap <= gap_minutes for gap in gaps]
    tail = 1
    for value in reversed(connected):
        if not value:
            break
        tail += 1
    return {"session_count": 1 + sum(not value for value in connected),
            "close_event_count": 1 + sum(connected),
            "trailing_session_event_count": tail}


def derive_session_threshold(trajectories, source, method):
    """Estimate minutes from positive observed training gaps, without outcomes.

    ``window_gaps`` retains overlapping-window multiplicity. ``unique_edges``
    counts each observed user/time/POI edge once. The method is an explicit part
    of the candidate contract, not selected within this function.
    """
    if source not in {"window_gaps", "unique_edges"}:
        raise ValueError("Unsupported training gap source: " + str(source))
    gaps, seen = [], set()
    for frame in trajectories:
        history = _history(frame)
        times = _times(history)
        events = [(int(row.user_id), str(time), int(row.poi_id))
                  for row, time in zip(history.itertuples(index=False), times)]
        for left, right, start, end in zip(events, events[1:], times, times[1:]):
            edge = (left, right)
            if source == "unique_edges" and edge in seen:
                continue
            seen.add(edge)
            gap = (end - start).total_seconds() / 60
            if np.isfinite(gap) and gap > 0:
                gaps.append(gap)
    if not gaps:
        raise ValueError("No positive observed training gaps")
    values = np.asarray(gaps, dtype=float)
    quantiles = {"quartile_1": .25, "median": .5, "quartile_3": .75, "decile_9": .9}
    if method in quantiles:
        return float(np.quantile(values, quantiles[method]))
    raise ValueError("Unsupported session threshold method: " + str(method))


def observed_event_keys(frame):
    history = _history(frame)
    return tuple((int(row.user_id), str(time), int(row.poi_id))
                 for row, time in zip(history.itertuples(index=False), _times(history)))


def eligible_trailing_session_multiplicities(trajectories, eligible, gap_minutes):
    eligible = np.asarray(eligible)
    if eligible.shape != (len(trajectories),) or not np.isin(eligible, (0, 1)).all():
        raise ValueError("Eligibility must be one boolean or 0/1 value per trajectory")
    keys = []
    for frame in trajectories:
        events = observed_event_keys(frame)
        tail = observed_session_counts(frame, gap_minutes)["trailing_session_event_count"]
        keys.append((events[0][0], events[-tail][1], events[-tail][2]))
    counts = Counter(key for key, keep in zip(keys, eligible) if keep)
    return np.asarray([counts[key] if keep else 0 for key, keep in zip(keys, eligible)], dtype=np.int64)


def observed_stream_session_representatives(trajectories, gap_minutes, eligible=None, selection="first"):
    """Select one stored window per session in the merged observed event stream.

    Deduplicate user/time/POI events across all observed prefixes, then sort each
    user's events chronologically. Equal-time events retain first appearance
    order. A gap strictly greater than the supplied training-derived threshold
    begins a session. Each window belongs to its last observed event's session.

    First/last means stored window order, not target time. Group construction
    uses all input prefixes; selection uses eligible windows only. Held-out rows
    are never read. The returned 0/1 weights do not duplicate predictions.
    """
    if not np.isfinite(gap_minutes) or gap_minutes < 0:
        raise ValueError("Session gap must be finite and nonnegative")
    if selection not in {"first", "last"}:
        raise ValueError("Session representative selection must be first or last")
    if eligible is None:
        eligible = np.ones(len(trajectories), dtype=bool)
    eligible = np.asarray(eligible)
    if eligible.shape != (len(trajectories),) or not np.isin(eligible, (0, 1)).all():
        raise ValueError("Eligibility must be one boolean or 0/1 value per trajectory")
    histories = [observed_event_keys(frame) for frame in trajectories]
    by_user = defaultdict(dict)
    for events in histories:
        for event in events:
            by_user[event[0]].setdefault(event, len(by_user[event[0]]))
    session_for_event = {}
    for events in by_user.values():
        ordered = sorted(events, key=lambda e: (pd.Timestamp(e[1]), events[e]))
        previous, start = None, None
        for event in ordered:
            stamp = pd.Timestamp(event[1])
            if previous is None or (stamp - previous).total_seconds() / 60 > gap_minutes:
                start = event
            session_for_event[event] = start
            previous = stamp
    weights = np.zeros(len(histories), dtype=np.int64)
    seen = set()
    order = range(len(histories)) if selection == "first" else reversed(range(len(histories)))
    for index in order:
        if not eligible[index]:
            continue
        key = session_for_event[histories[index][-1]]
        if key not in seen:
            weights[index] = 1
            seen.add(key)
    return weights


def novelty_signature_join_multiplicities(training, trajectories, eligible=None):
    """Join windows sharing an unseen-transition set and multiple novelty runs.

    Training includes its supervised final check-ins. Evaluation uses observed
    prefixes only. A novelty run is a maximal consecutive sequence of observed
    transitions absent from training. Keep windows with at least two such runs,
    then self-join eligible windows on their sorted set of unseen POI edges.
    Every retained window's multiplicity is the actual size of its signature
    group. Neither test targets nor correctness enter the construction.
    """
    if eligible is None:
        eligible = np.ones(len(trajectories), dtype=bool)
    eligible = np.asarray(eligible)
    if eligible.shape != (len(trajectories),) or not np.isin(eligible, (0, 1)).all():
        raise ValueError("Eligibility must be one boolean or 0/1 value per trajectory")
    known = set()
    for frame in training:
        ids = list(map(int, frame.poi_id))
        known.update(zip(ids, ids[1:]))
    signatures, selected = [], []
    for frame, keep in zip(trajectories, eligible):
        ids = list(map(int, _history(frame).poi_id))
        edges = list(zip(ids, ids[1:]))
        unseen = [edge not in known for edge in edges]
        runs = sum(flag and (index == 0 or not unseen[index - 1]) for index, flag in enumerate(unseen))
        signatures.append(tuple(sorted({edge for edge, flag in zip(edges, unseen) if flag})))
        selected.append(bool(keep) and runs > 1)
    counts = Counter(signature for signature, keep in zip(signatures, selected) if keep)
    return np.asarray([counts[signature] if keep else 0 for signature, keep in zip(signatures, selected)], dtype=np.int64)


def observed_history_quality(frame):
    """Duration and maximum spatial step of the observed prefix only."""
    from .features import _finite_coordinate, _haversine_km, _timestamps

    history = _history(frame)
    timestamps = _timestamps(history)
    if any(pd.isna(stamp) for stamp in timestamps):
        raise ValueError("Missing observed timestamp in quality envelope")
    coordinates = list(zip(history.latitude, history.longitude))
    if not all(_finite_coordinate(*point) for point in coordinates):
        raise ValueError("Invalid observed coordinate in quality envelope")
    distances = [_haversine_km(a, b) for a, b in zip(coordinates, coordinates[1:])]
    return {"span_hours": (timestamps[-1] - timestamps[0]).total_seconds() / 3600,
            "step_distance_km_max": max(distances, default=0.)}


def derive_history_quality_envelope(training):
    """Training-only nonnegative Tukey bounds, retaining window multiplicity."""
    if not training:
        raise ValueError("Quality calibration requires training trajectories")
    records = [observed_history_quality(frame) for frame in training]
    bounds = {}
    for feature in ("span_hours", "step_distance_km_max"):
        values = np.asarray([record[feature] for record in records])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite training quality feature")
        q1, q3 = map(float, np.quantile(values, [.25, .75]))
        bounds[feature] = {"q1": q1, "q3": q3,
                           "lower": max(0., q1 - 1.5 * (q3 - q1)),
                           "upper": q3 + 1.5 * (q3 - q1)}
    return bounds


def history_quality_mask(trajectories, bounds):
    """Use a frozen training envelope; never calibrate on evaluation outcomes."""
    if set(bounds) != {"span_hours", "step_distance_km_max"}:
        raise ValueError("Quality envelope requires duration and maximum step")
    for interval in bounds.values():
        if not np.isfinite([interval["lower"], interval["upper"]]).all() or interval["lower"] > interval["upper"]:
            raise ValueError("Invalid quality envelope interval")
    return np.asarray([all(bounds[key]["lower"] <= value <= bounds[key]["upper"]
                           for key, value in observed_history_quality(frame).items())
                       for frame in trajectories], dtype=bool)
