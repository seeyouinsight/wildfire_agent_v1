import json
import ee
from typing import Dict, Any

from agent.config import SENSOR_COLLECTION_MAP
from agent.utils.gee_init import init_gee
from agent.utils.gee_common import load_region, get_collection, safe_count


def run_quick_fire_detection(
    region_geojson: str,
    date_start: str,
    date_end: str,
    sensor: str = "VIIRS",
) -> Dict[str, Any]:
    try:
        init_gee()

        sensor = sensor.strip()
        if sensor not in ["MODIS", "VIIRS"]:
            raise ValueError("run_quick_fire_detection currently supports only MODIS or VIIRS.")

        region = load_region(region_geojson)
        collection = get_collection(sensor).filterDate(date_start, date_end).filterBounds(region)

        image_count = safe_count(collection)
        if image_count == 0:
            return {
                "status": "ok",
                "workflow_id": "modis_viirs_quick_active_fire",
                "sensor": sensor,
                "collection_id": SENSOR_COLLECTION_MAP[sensor]["collection_id"],
                "date_start": date_start,
                "date_end": date_end,
                "image_count": 0,
                "fire_pixel_count_estimate": 0,
                "message": "No images found for this region and date range.",
            }

        first_img = ee.Image(collection.first())
        band_names = first_img.bandNames().getInfo()

        if "FireMask" not in band_names:
            return {
                "status": "error",
                "workflow_id": "modis_viirs_quick_active_fire",
                "sensor": sensor,
                "collection_id": SENSOR_COLLECTION_MAP[sensor]["collection_id"],
                "message": f"Band 'FireMask' not found. Available bands: {band_names}",
            }

        fire_mask_collection = collection.map(
            lambda img: ee.Image(img).select("FireMask").gt(0).rename("fire_binary")
        )
        fire_sum = ee.ImageCollection(fire_mask_collection).sum().clip(region)

        fire_pixel_count = fire_sum.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=1000,
            maxPixels=1e9,
        ).get("fire_binary")

        try:
            fire_pixel_count_value = int(ee.Number(fire_pixel_count).getInfo())
        except Exception:
            fire_pixel_count_value = 0

        return {
            "status": "ok",
            "workflow_id": "modis_viirs_quick_active_fire",
            "sensor": sensor,
            "collection_id": SENSOR_COLLECTION_MAP[sensor]["collection_id"],
            "date_start": date_start,
            "date_end": date_end,
            "image_count": image_count,
            "fire_pixel_count_estimate": fire_pixel_count_value,
            "message": "Quick fire detection summary completed.",
        }

    except Exception as e:
        return {
            "status": "error",
            "workflow_id": "modis_viirs_quick_active_fire",
            "sensor": sensor,
            "message": str(e),
        }


if __name__ == "__main__":
    tokyo_region = {
        "type": "Polygon",
        "coordinates": [[[139.5, 35.4], [140.1, 35.4], [140.1, 35.9], [139.5, 35.9], [139.5, 35.4]]]
    }

    result = run_quick_fire_detection(
        region_geojson=json.dumps(tokyo_region, ensure_ascii=False),
        date_start="2024-08-01",
        date_end="2024-08-31",
        sensor="VIIRS",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))