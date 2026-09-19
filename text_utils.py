import re
import time
import pickle as pkl
import math
from typing import Optional
import pandas as pd


PROMPT_CONTRACTS = ("coordinate_free_v1", "coordinate_pairs_v2")
COORDINATE_MISSING_POLICIES = ("error", "omit")


def _safe_time_fmt(ts) -> str:
    """
    Cross-platform-ish time formatting.
    Tries %-I (Unix), falls back to %I (Windows) without leading-zero removal.
    """
    try:
        return ts.strftime("%B %d, %Y, at %-I:%M %p")
    except ValueError:
        return ts.strftime("%B %d, %Y, at %I:%M %p")

def _get_category_col(df: pd.DataFrame):
    """
    Prefer NYC column name if present, otherwise generic 'category'.
    Return None if neither exists.
    """
    if "poi_category_name" in df.columns:
        return "poi_category_name"
    if "category" in df.columns:
        return "category"
    return None


def convert_latlon_to_text(lat: float, lon: float) -> str:
    """
    Use geopy Nominatim to reverse-geocode (lat, lon) into an address.
    Returns a string with city, county, road, house_number, and a generic category.
    """
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="geo_converter")
    location = geolocator.reverse((lat, lon), exactly_one=True)

    if location and location.raw.get("address"):
        address = location.raw["address"]
        city = address.get("city", address.get("town", address.get("village", "Unknown City")))
        county = address.get("county", "Unknown County")
        road = address.get("road", "Unknown Road")
        house_number = address.get("house_number", "Unknown Number")
        category = address.get("amenity", "General Area")
        return f"{city} {county} {road} {house_number} ({category})"
    else:
        return "Address not found"


def build_coordinates2addresses(checkins_df: pd.DataFrame, out_path: str, throttle_s: float = 1.0):
    """
    For all unique (lat, lon) pairs in checkins_df,
    build a dict mapping (lat, lon) -> textual address.
    If a category column exists, replace the parentheses content with the POI category.
    NOTE: Nominatim usage policy discourages aggressive batching; throttle requests.
    """
    from tqdm import tqdm

    unique_coords = set(zip(checkins_df['latitude'], checkins_df['longitude']))
    location2address = {}
    cat_col = _get_category_col(checkins_df)

    for coords in tqdm(unique_coords, desc="Building coords->address map"):
        lat, lon = coords
        textual_address = convert_latlon_to_text(lat, lon)

        if cat_col is not None:
            subset = checkins_df[(checkins_df['latitude'] == lat) & (checkins_df['longitude'] == lon)]
            if len(subset) > 0:
                poi_cat = subset.iloc[0][cat_col]
                if pd.notna(poi_cat):
                    textual_address = re.sub(r"\([^)]*\)", f"({poi_cat})", textual_address)

        location2address[coords] = textual_address
        if throttle_s:
            time.sleep(throttle_s)  # Reduce latency from geocoding service

    with open(out_path, 'wb') as f:
        pkl.dump(location2address, f)
    return location2address


def load_coordinates2addresses(path: str) -> dict:
    """Load the pickled dictionary of (lat, lon) -> address."""
    with open(path, 'rb') as f:
        return pkl.load(f)


def format_row_coordinate(
    row: pd.Series,
    *,
    precision: int,
    missing: str,
) -> Optional[str]:
    """Return the coordinate pair stored on one trajectory row.

    ``coordinate_pairs_v2`` deliberately serializes the row values directly;
    it must never collapse repeated POIs through a catalogue-level lookup.
    """

    if missing not in COORDINATE_MISSING_POLICIES:
        raise ValueError(f"Unknown coordinate missing-data policy: {missing}")
    if precision < 0 or precision > 12:
        raise ValueError("Coordinate precision must be between 0 and 12")
    try:
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
    except (KeyError, TypeError, ValueError):
        if missing == "omit":
            return None
        raise ValueError("Observed row is missing valid latitude/longitude values")
    valid = (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90.0 <= latitude <= 90.0
        and -180.0 <= longitude <= 180.0
    )
    if not valid:
        if missing == "omit":
            return None
        raise ValueError(
            f"Observed row has invalid coordinates: {latitude}, {longitude}"
        )
    return f"<{latitude:.{precision}f}, {longitude:.{precision}f}>"


def build_prompt(
    traj: pd.DataFrame,
    userid: int,
    poi_id_range: int,
    location2address: dict = None,
    *,
    prompt_contract: str = "coordinate_free_v1",
    coordinate_precision: int = 6,
    coordinate_missing: str = "error",
) -> str:
    """
    Build a textual prompt describing all but the last check-in as context,
    and the last check-in as the final question + answer block.

    - Uses 'local_time' if present, else falls back to 'event_time_utc'.
    - Uses NYC 'poi_category_name' or generic 'category' if available; otherwise omits.
    - ``coordinate_free_v1`` retains the historical coordinate-free text.
    - ``coordinate_pairs_v2`` appends each observed row's own coordinate pair.
    - Note: We don't inject address text into the prompt (keeps parity with the paper).
    """
    if prompt_contract not in PROMPT_CONTRACTS:
        raise ValueError(f"Unknown prompt contract: {prompt_contract}")
    if coordinate_missing not in COORDINATE_MISSING_POLICIES:
        raise ValueError(
            f"Unknown coordinate missing-data policy: {coordinate_missing}"
        )
    if coordinate_precision < 0 or coordinate_precision > 12:
        raise ValueError("Coordinate precision must be between 0 and 12")
    location2address = location2address or {}
    cat_col = _get_category_col(traj)

    # select a time column for display
    time_col = "local_time" if "local_time" in traj.columns else (
        "event_time_utc" if "event_time_utc" in traj.columns else None
    )
    if time_col is None:
        raise KeyError("Trajectory is missing both 'local_time' and 'event_time_utc'.")

    partial_traj = traj.iloc[:-1]
    final_checkin = traj.iloc[-1]

    # Summaries for partial trajectory
    lines = []
    for _, row in partial_traj.iterrows():
        poi_id = row.poi_id
        tstamp = _safe_time_fmt(row[time_col])
        coordinate_text = None
        if prompt_contract == "coordinate_pairs_v2":
            coordinate_text = format_row_coordinate(
                row,
                precision=coordinate_precision,
                missing=coordinate_missing,
            )
        location_clause = f" at {coordinate_text}" if coordinate_text else ""
        if cat_col and pd.notna(row.get(cat_col)):
            cat_text = str(row[cat_col])
            if prompt_contract == "coordinate_pairs_v2" and coordinate_text:
                lines.append(
                    f"At {tstamp}, user {userid} visited POI id {poi_id}"
                    f"{location_clause}, which is a {cat_text}."
                )
            else:
                lines.append(
                    f"At {tstamp}, user {userid} visited POI id {poi_id}"
                    f"{location_clause} which is a {cat_text}."
                )
        else:
            lines.append(
                f"At {tstamp}, user {userid} visited POI id {poi_id}"
                f"{location_clause}."
            )

    trajectory_str = "\n".join(lines)
    # IMPORTANT: poi_id_range is the COUNT (max_id + 1). Display 0..(count-1).
    max_id_inclusive = max(0, poi_id_range - 1)

    question = (
        f"<question>: The following is a trajectory of user {userid}:\n"
        f"{trajectory_str}\n\n"
        f"Given the data, which POI id will user {userid} visit next within 24 hours?\n"
        f"Note that POI id is an integer in the range from 0 to {max_id_inclusive}."
    )
    # Keep the held-out timestamp out of the autoregressive answer prefix: under
    # teacher forcing it would become an input when predicting the POI token.
    answer = f"<answer>: POI id {final_checkin.poi_id}."
    return question + "\n" + answer


def inject_pre_cacl_safety_scores_to_prompt(
    prompt_text: str,
    safety_scores: list[float],
) -> str:
    """
    Inject safety lines after each 'At ... visited POI id X' line.
    Also modifies 'Given the data,' to reference 'route safety scores'.
    """
    lines = prompt_text.split("\n")
    new_lines = []
    poi_pattern = re.compile(r"visited POI id\s+(\d+)(?:\s+|(?=[.,!?]|$))")

    score_index = 0
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("At ") and "visited POI id" in line_stripped:
            new_lines.append(line)
            match = poi_pattern.search(line_stripped)
            if match and score_index < len(safety_scores):
                poi_id = match.group(1)
                score = safety_scores[score_index]
                score_index += 1
                new_lines.append(f"The safety score from POI {poi_id} to the next POI is {round(score, 3)}.")
        elif line_stripped.startswith("Given the data,"):
            replaced = line_stripped.replace(
                "Given the data,",
                "Given the data (including the route safety scores),"
            )
            new_lines.append(replaced)
            new_lines.append(
                "Please consider the user’s trajectory and these safety scores when determining the most likely POI."
            )
        else:
            new_lines.append(line)

    return "\n".join(new_lines)
