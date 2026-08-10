import json
import ee
from agent.config import SENSOR_COLLECTION_MAP


def load_region(region_geojson: str) -> ee.Geometry:
    try:
        geojson_obj = json.loads(region_geojson)
    except Exception as e:
        raise ValueError(f"Invalid region_geojson JSON: {e}")

    if isinstance(geojson_obj, dict) and geojson_obj.get("type") == "Feature":
        geometry = geojson_obj.get("geometry")
    else:
        geometry = geojson_obj

    if not isinstance(geometry, dict) or "type" not in geometry:
        raise ValueError("GeoJSON must contain a valid geometry object")

    return ee.Geometry(geometry)


def get_collection(sensor: str) -> ee.ImageCollection:
    sensor = sensor.strip()
    if sensor not in SENSOR_COLLECTION_MAP:
        raise ValueError(
            f"Unsupported sensor: {sensor}. Supported sensors: {list(SENSOR_COLLECTION_MAP.keys())}"
        )
    return ee.ImageCollection(SENSOR_COLLECTION_MAP[sensor]["collection_id"])


def safe_count(col: ee.ImageCollection) -> int:
    try:
        return int(col.size().getInfo())
    except Exception:
        return 0