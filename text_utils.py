from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import pickle
import pickle as pkl
import re
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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
    - ``coordinate_free_v1`` retains the coordinate-free prompt text.
    - ``coordinate_pairs_v2`` appends each observed row's own coordinate pair.
    - Address text is not included in the configured prompt.
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

def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload

PROMPT_CONTRACT = "coordinate_pairs_v2"
SERIALIZER_VERSION = "coordinate_pairs_v2.inline_row_coordinates.v1"


ANSWER_MARKER = "<answer>:"
SAFETY_LINE = re.compile(
    r"The safety score from POI \d+ to the next POI is [-0-9.]+\."
)
HISTORY_LINE = re.compile(r"^At .+ visited POI id \d+", re.MULTILINE)


@dataclass(frozen=True)
class PromptAudit:
    count: int
    observed_row_counts: dict[int, int]
    coordinate_pair_counts: dict[int, int]
    safety_line_counts: dict[int, int]
    row_coordinate_mismatches: int
    target_coordinate_leaks: int
    target_timestamp_leaks: int
    answer_schema_violations: int
    safety_transition_mismatches: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def validate_manifest_header(manifest: Mapping[str, object]) -> None:
    """Reject prompt manifests created for another spatial contract."""

    expected = {
        "prompt_contract": PROMPT_CONTRACT,
        "serializer_version": SERIALIZER_VERSION,
        "coordinate_source": "row_level_numeric_trajectory",
        "coordinate_precision": 6,
        "coordinate_missing": "error",
        "target_fields_in_model_input": [],
        "answer_format": "poi_only",
        "poi_id_range_source": "train_only",
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Prompt manifest is not coordinate_pairs_v2: {mismatches}")


def load_prompt_manifest(numeric_root: str | Path) -> dict:
    manifest_path = Path(numeric_root).resolve() / "coordinate_pairs_v2_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a JSON object: {manifest_path}")
    validate_manifest_header(manifest)
    return manifest


def _time_column(trajectory) -> str:
    if "local_time" in trajectory.columns:
        return "local_time"
    if "event_time_utc" in trajectory.columns:
        return "event_time_utc"
    raise KeyError("Trajectory is missing both local_time and event_time_utc")


def _atomic_json(path: Path, value: object, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                indent=indent,
                sort_keys=indent is not None,
            )
            if indent is not None:
                stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def audit_prompts(
    texts: Sequence[str],
    trajectories: Sequence,
    *,
    include_safety: bool,
    coordinate_precision: int = 6,
    coordinate_missing: str = "error",
) -> PromptAudit:
    """Fail closed on coordinate association, target leakage, or safety leakage."""

    if len(texts) != len(trajectories):
        raise ValueError(f"Prompt/trajectory count mismatch: {len(texts)} != {len(trajectories)}")
    observed_counts: Counter[int] = Counter()
    coordinate_counts: Counter[int] = Counter()
    safety_counts: Counter[int] = Counter()
    row_mismatches = target_coordinate_leaks = target_timestamp_leaks = 0
    answer_violations = safety_mismatches = 0

    for text, trajectory in zip(texts, trajectories):
        if len(trajectory) < 2:
            raise ValueError("A coordinate_pairs_v2 trajectory needs history and target rows")
        if ANSWER_MARKER not in text:
            raise ValueError("Prompt has no answer marker")
        question, raw_answer = text.split(ANSWER_MARKER, 1)
        observed = trajectory.iloc[:-1]
        target = trajectory.iloc[-1]
        history_lines = [
            line
            for line in question.split("\n")
            if line.startswith("At ") and " visited POI id " in line
        ]
        observed_counts[len(history_lines)] += 1
        if len(history_lines) != len(observed):
            row_mismatches += 1

        observed_coordinates: list[str] = []
        for offset, (_, row) in enumerate(observed.iterrows()):
            coordinate = format_row_coordinate(
                row, precision=coordinate_precision, missing=coordinate_missing
            )
            if coordinate is not None:
                observed_coordinates.append(coordinate)
            if offset >= len(history_lines):
                continue
            line = history_lines[offset]
            if f"visited POI id {row.poi_id}" not in line:
                row_mismatches += 1
            if coordinate is None:
                if re.search(r" at <-?\d", line):
                    row_mismatches += 1
            elif f" at {coordinate}" not in line:
                row_mismatches += 1
        coordinate_counts[sum(bool(re.search(r" at <-?\d", line)) for line in history_lines)] += 1

        target_coordinate = format_row_coordinate(
            target, precision=coordinate_precision, missing="omit"
        )
        if (
            target_coordinate is not None
            and target_coordinate not in observed_coordinates
            and target_coordinate in question
        ):
            target_coordinate_leaks += 1

        time_column = _time_column(trajectory)
        target_timestamp = _safe_time_fmt(target[time_column])
        observed_timestamps = {
            _safe_time_fmt(row[time_column]) for _, row in observed.iterrows()
        }
        if target_timestamp not in observed_timestamps and target_timestamp in question:
            target_timestamp_leaks += 1

        if f"{ANSWER_MARKER}{raw_answer}".strip() != f"{ANSWER_MARKER} POI id {target.poi_id}.":
            answer_violations += 1

        safety_lines = [line for line in question.split("\n") if SAFETY_LINE.fullmatch(line)]
        safety_counts[len(safety_lines)] += 1
        expected_safety: list[str] = []
        if include_safety:
            if "normalized_safety" not in trajectory.columns:
                safety_mismatches += 1
            else:
                for offset in range(max(0, len(trajectory) - 2)):
                    row = trajectory.iloc[offset]
                    expected_safety.append(
                        f"The safety score from POI {row.poi_id} to the next POI is "
                        f"{round(row.normalized_safety, 3)}."
                    )
        if safety_lines != expected_safety:
            safety_mismatches += 1

    audit = PromptAudit(
        count=len(texts),
        observed_row_counts=dict(sorted(observed_counts.items())),
        coordinate_pair_counts=dict(sorted(coordinate_counts.items())),
        safety_line_counts=dict(sorted(safety_counts.items())),
        row_coordinate_mismatches=row_mismatches,
        target_coordinate_leaks=target_coordinate_leaks,
        target_timestamp_leaks=target_timestamp_leaks,
        answer_schema_violations=answer_violations,
        safety_transition_mismatches=safety_mismatches,
    )
    if any(
        (
            audit.row_coordinate_mismatches,
            audit.target_coordinate_leaks,
            audit.target_timestamp_leaks,
            audit.answer_schema_violations,
            audit.safety_transition_mismatches,
        )
    ):
        raise ValueError(f"coordinate_pairs_v2 prompt audit failed: {audit}")
    return audit


def materialize_prompts(
    numeric_root: str | Path,
    *,
    coordinate_precision: int = 6,
    coordinate_missing: str = "error",
) -> tuple[dict[str, dict[str, list[str]]], dict]:
    """Materialize both prompt variants in memory and return their manifest."""

    numeric_root = Path(numeric_root).resolve()
    splits = ("train", "validation", "test")
    numeric_paths = {split: numeric_root / f"{split}_trajectories.pickle" for split in splits}
    safety_paths = {
        split: numeric_root / "safety" / f"{split}_trajs_with_safety.pickle"
        for split in splits
    }
    source_paths = {
        **{f"numeric_{name}": path for name, path in numeric_paths.items()},
        **{f"safety_{name}": path for name, path in safety_paths.items()},
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing numeric prompt inputs: " + ", ".join(missing))
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}

    numeric: dict[str, list] = {}
    scored: dict[str, list] = {}
    source_audit: dict[str, dict] = {}
    for split in splits:
        with numeric_paths[split].open("rb") as handle:
            numeric[split] = pickle.load(handle)
        with safety_paths[split].open("rb") as handle:
            scored[split] = pickle.load(handle)
        if len(numeric[split]) != len(scored[split]):
            raise ValueError(f"Numeric/safety trajectory count mismatch for {split}")
        for index, (plain, safety) in enumerate(zip(numeric[split], scored[split])):
            if len(plain) != len(safety) or not plain.equals(safety[plain.columns]):
                raise ValueError(f"Numeric/safety row mismatch for {split} trajectory {index}")
        lengths = Counter(len(trajectory) for trajectory in numeric[split])
        coordinate_valid_rows = 0
        coordinate_invalid_rows = 0
        for trajectory in numeric[split]:
            for _, row in trajectory.iterrows():
                coordinate = format_row_coordinate(
                    row, precision=coordinate_precision, missing="omit"
                )
                if coordinate is None:
                    coordinate_invalid_rows += 1
                else:
                    coordinate_valid_rows += 1
        source_audit[split] = {
            "trajectory_count": len(numeric[split]),
            "trajectory_length_counts": {
                str(length): count for length, count in sorted(lengths.items())
            },
            "row_count": coordinate_valid_rows + coordinate_invalid_rows,
            "coordinate_valid_rows": coordinate_valid_rows,
            "coordinate_invalid_rows": coordinate_invalid_rows,
            "numeric_safety_alignment": True,
        }

    train_pois = [
        int(value)
        for trajectory in numeric["train"]
        for value in trajectory["poi_id"].tolist()
    ]
    poi_id_range = max(train_pois) + 1 if train_pois else 0
    materialized: dict[str, dict[str, list[str]]] = {}
    variants: dict[str, dict] = {}
    for variant, include_safety in (("no_safety", False), ("with_safety", True)):
        materialized[variant] = {}
        split_manifest: dict[str, dict] = {}
        for split in splits:
            trajectories = scored[split] if include_safety else numeric[split]
            texts: list[str] = []
            coordinate_free_controls: list[str] = []
            for trajectory in trajectories:
                text = build_prompt(
                    trajectory,
                    userid=trajectory["user_id"].iloc[0],
                    poi_id_range=poi_id_range,
                    prompt_contract=PROMPT_CONTRACT,
                    coordinate_precision=coordinate_precision,
                    coordinate_missing=coordinate_missing,
                )
                if include_safety:
                    scores = list(
                        trajectory["normalized_safety"].iloc[: max(0, len(trajectory) - 2)]
                    )
                    text = inject_pre_cacl_safety_scores_to_prompt(text, scores)
                texts.append(text)
                control = build_prompt(
                    trajectory,
                    userid=trajectory["user_id"].iloc[0],
                    poi_id_range=poi_id_range,
                    prompt_contract="coordinate_free_v1",
                )
                if include_safety:
                    control = inject_pre_cacl_safety_scores_to_prompt(
                        control, scores
                    )
                coordinate_free_controls.append(control)
            audit = audit_prompts(
                texts,
                trajectories,
                include_safety=include_safety,
                coordinate_precision=coordinate_precision,
                coordinate_missing=coordinate_missing,
            )
            materialized[variant][split] = texts
            coordinate_pairs_hash = sequence_sha256(texts)
            coordinate_free_hash = sequence_sha256(coordinate_free_controls)
            if coordinate_pairs_hash == coordinate_free_hash:
                raise ValueError(
                    f"{variant} {split} prompt hash did not change from coordinate_free_v1"
                )
            split_manifest[split] = {
                "count": len(texts),
                "sequence_sha256": coordinate_pairs_hash,
                "coordinate_free_v1_control_sequence_sha256": coordinate_free_hash,
                "audit": asdict(audit),
            }
        variants[variant] = {
            "safety_contract": "observed_transitions_only" if include_safety else "no_numeric_safety",
            "splits": split_manifest,
        }
    manifest = {
        "prompt_contract": PROMPT_CONTRACT,
        "serializer_version": SERIALIZER_VERSION,
        "coordinate_source": "row_level_numeric_trajectory",
        "coordinate_precision": coordinate_precision,
        "coordinate_missing": coordinate_missing,
        "target_fields_in_model_input": [],
        "answer_format": "poi_only",
        "poi_id_range_source": "train_only",
        "poi_id_range": poi_id_range,
        "source_sha256": source_hashes,
        "source_audit": source_audit,
        "variants": variants,
    }
    return materialized, manifest


def write_canonical_prompts(
    numeric_root: str | Path,
    *,
    output_root: str | Path | None = None,
    coordinate_precision: int = 6,
    coordinate_missing: str = "error",
) -> dict:
    """Atomically replace the six canonical derived prompt files after auditing all."""

    numeric_root = Path(numeric_root).resolve()
    materialized, manifest = materialize_prompts(
        numeric_root,
        coordinate_precision=coordinate_precision,
        coordinate_missing=coordinate_missing,
    )
    output_root = Path(output_root).resolve() if output_root is not None else numeric_root
    paths = {
        "no_safety": {
            split: output_root / f"textual_{split}_trajs.json"
            for split in ("train", "validation", "test")
        },
        "with_safety": {
            split: output_root / "safety" / f"safety_textual_{split}_trajs.json"
            for split in ("train", "validation", "test")
        },
    }
    for variant, split_paths in paths.items():
        for split, path in split_paths.items():
            _atomic_json(path, materialized[variant][split])
            manifest["variants"][variant]["splits"][split]["file_sha256"] = sha256_file(path)
            manifest["variants"][variant]["splits"][split]["path"] = str(path.relative_to(output_root))
    manifest_path = output_root / "coordinate_pairs_v2_manifest.json"
    _atomic_json(manifest_path, manifest, indent=2)
    return manifest


def verify_canonical_prompts(numeric_root: str | Path) -> dict:
    """Regenerate in memory and verify every tracked canonical prompt byte-for-byte."""

    numeric_root = Path(numeric_root).resolve()
    materialized, expected = materialize_prompts(numeric_root)
    manifest_path = numeric_root / "coordinate_pairs_v2_manifest.json"
    actual = load_prompt_manifest(numeric_root)
    for variant, prefix in (("no_safety", ""), ("with_safety", "safety/")):
        for split in ("train", "validation", "test"):
            name = f"textual_{split}_trajs.json" if variant == "no_safety" else f"safety_textual_{split}_trajs.json"
            path = numeric_root / prefix / name
            values = json.loads(path.read_text(encoding="utf-8"))
            if values != materialized[variant][split]:
                raise ValueError(f"Canonical prompt content differs from regeneration: {path}")
            metadata = actual["variants"][variant]["splits"][split]
            expected_metadata = expected["variants"][variant]["splits"][split]
            if metadata["sequence_sha256"] != expected_metadata["sequence_sha256"]:
                raise ValueError(f"Prompt sequence hash differs: {path}")
            if metadata["file_sha256"] != sha256_file(path):
                raise ValueError(f"Prompt file hash differs: {path}")
            expected_metadata["file_sha256"] = sha256_file(path)
            expected_metadata["path"] = str(path.relative_to(numeric_root))
    # JSON converts integer audit-histogram keys to strings. Compare the
    # regenerated manifest in the same representation as the stored one.
    if actual != json.loads(json.dumps(expected)):
        raise ValueError("Canonical prompt manifest differs from regeneration")
    return actual
