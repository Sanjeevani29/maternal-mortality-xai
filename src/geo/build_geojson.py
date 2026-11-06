"""Simplify district boundaries for fast web rendering and key them to the NFHS-5 district frame.

``mapshaper`` is used when it is already installed locally (``mapshaper`` on PATH); otherwise a
pure-Python Douglas-Peucker simplifier with coordinate rounding is used. Polygons are joined to
NFHS districts by normalised state/district names (the community GeoJSON's census codes are not
reliable). Districts created after the 2011-era boundary file stay unmapped and are listed in the
geo audit; no geometry is fabricated.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src.data.merge_nfhs_hmis import load_aliases, normalise_district, normalise_state

GEO_STATE_ALIASES = {
    "arunanchal pradesh": "arunachal pradesh",
    "dadara and nagar havelli": "dadra and nagar haveli",
}
# Boundary-file spellings -> normalised NFHS names (after NFHS aliases are applied).
GEO_DISTRICT_ALIASES = {
    "bauda": "baudh",
    "dohad": "dahod",
    "dakshin bastar dantewada": "dantewada",
    "east nimar": "khandwa",
    "west nimar": "khargone",
    "kansiram nagar": "kashi ram nagar",
    "north 24 parganas": "north twenty four parganas",
    "south 24 parganas": "south twenty four parganas",
    "sant ravi das nagar bhadohi": "sant ravidas nagar",
    "saran chhapra": "saran",
}


def district_key(state: object, district: object) -> str:
    return f"{str(state).strip()}|{str(district).strip()}"


def _perpendicular_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    if np.allclose(start, end):
        return np.linalg.norm(points - start, axis=1)
    direction = end - start
    return np.abs(np.cross(direction, points - start)) / np.linalg.norm(direction)


def douglas_peucker(coordinates: np.ndarray, tolerance: float) -> np.ndarray:
    """Iterative Douglas-Peucker line simplification."""
    if len(coordinates) < 3:
        return coordinates
    keep = np.zeros(len(coordinates), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(coordinates) - 1)]
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        distances = _perpendicular_distance(coordinates[first + 1 : last], coordinates[first], coordinates[last])
        index = int(np.argmax(distances))
        if distances[index] > tolerance:
            split = first + 1 + index
            keep[split] = True
            stack.extend([(first, split), (split, last)])
    return coordinates[keep]


def _simplify_ring(ring: list, tolerance: float, digits: int) -> list:
    array = np.asarray(ring, dtype=float)[:, :2]
    simplified = douglas_peucker(array, tolerance)
    if len(simplified) < 4:  # keep shapes: fall back to the original ring for tiny polygons
        simplified = array if len(array) < 8 else array[:: max(1, len(array) // 6)]
        simplified = np.vstack([simplified, simplified[:1]]) if not np.allclose(simplified[0], simplified[-1]) else simplified
    return np.round(simplified, digits).tolist()


def simplify_geometry(geometry: dict, tolerance: float = 0.01, digits: int = 4) -> dict:
    if geometry["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [_simplify_ring(ring, tolerance, digits) for ring in geometry["coordinates"]]}
    if geometry["type"] == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [[_simplify_ring(ring, tolerance, digits) for ring in polygon] for polygon in geometry["coordinates"]],
        }
    return geometry


def _simplify_with_mapshaper(source: Path) -> dict | None:
    executable = shutil.which("mapshaper")
    if executable is None:
        return None
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "simplified.geojson"
        completed = subprocess.run(
            [executable, str(source), "-simplify", "dp", "8%", "keep-shapes", "-o", "format=geojson", str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not output.exists():
            return None
        return json.loads(output.read_text(encoding="utf-8"))


def build_simplified_geojson(
    source: Path,
    districts: pd.DataFrame,
    output: Path,
    audit_path: Path,
    aliases_path: Path,
    tolerance: float = 0.01,
) -> dict[str, object]:
    collection = _simplify_with_mapshaper(source)
    method = "mapshaper dp 8% keep-shapes"
    if collection is None:
        collection = json.loads(source.read_text(encoding="utf-8"))
        for feature in collection["features"]:
            feature["geometry"] = simplify_geometry(feature["geometry"], tolerance)
        method = f"python douglas-peucker tolerance={tolerance} deg, 4-decimal rounding"

    aliases = load_aliases(aliases_path)
    aliases["states"].update(GEO_STATE_ALIASES)
    aliases["districts"].update(GEO_DISTRICT_ALIASES)
    targets = districts[["state", "district"]].drop_duplicates().copy()
    targets["_state"] = targets["state"].map(lambda value: normalise_state(value, aliases))
    targets["_district"] = targets["district"].map(lambda value: normalise_district(value, aliases))
    by_state = {state: group for state, group in targets.groupby("_state")}
    used: set[str] = set()
    audit: list[dict[str, object]] = []
    kept_features = []
    for feature in collection["features"]:
        properties = feature.get("properties", {})
        name = normalise_district(properties.get("Dist_name"), aliases)
        state = normalise_state(properties.get("ST_NM"), aliases)
        state = aliases["hmis_state_overrides"].get(name, state)  # Leh/Kargil now belong to Ladakh
        candidates = by_state.get(state)
        key, method_used, score = None, "unmatched", None
        if candidates is not None:
            free = candidates.loc[~candidates.apply(lambda r: district_key(r["state"], r["district"]) in used, axis=1)]
            exact = free.loc[free["_district"] == name]
            if len(exact) == 1:
                match, method_used, score = exact.iloc[0], "exact", 100.0
            elif len(free):
                best = process.extractOne(name, dict(zip(free.index, free["_district"])), scorer=fuzz.token_sort_ratio)
                match, score = (free.loc[best[2]], best[1]) if best else (None, None)
                method_used = "fuzzy" if best and best[1] >= 85 else "unmatched"
                match = match if method_used == "fuzzy" else None
            else:
                match = None
            if method_used != "unmatched" and match is not None:
                key = district_key(match["state"], match["district"])
                used.add(key)
        audit.append(
            {
                "geo_state": properties.get("ST_NM"),
                "geo_district": properties.get("Dist_name"),
                "district_key": key,
                "method": method_used,
                "score": score,
            }
        )
        if key is not None:
            feature["id"] = key
            feature["properties"] = {"district_key": key, "geo_name": properties.get("Dist_name")}
            kept_features.append(feature)
    for _, row in targets.iterrows():
        key = district_key(row["state"], row["district"])
        if key not in used:
            audit.append({"geo_state": None, "geo_district": None, "district_key": key, "method": "no_polygon", "score": None})

    output.parent.mkdir(parents=True, exist_ok=True)
    simplified = {"type": "FeatureCollection", "features": kept_features}
    output.write_text(json.dumps(simplified, separators=(",", ":")), encoding="utf-8")
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    return {
        "method": method,
        "source_mb": round(source.stat().st_size / 1e6, 2),
        "output_mb": round(output.stat().st_size / 1e6, 2),
        "polygons_mapped": len(kept_features),
        "districts_without_polygon": int(sum(row["method"] == "no_polygon" for row in audit)),
    }
