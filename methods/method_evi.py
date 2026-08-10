import ee
from typing import Dict, Any

from agent.utils.gee_init import init_gee
from agent.utils.gee_common import load_region, get_collection


def _prep_landsat_evi_image(img: ee.Image) -> ee.Image:
    qa = img.select("QA_PIXEL")
    cloud_shadow = qa.bitwiseAnd(1 << 4).neq(0)
    cloud = qa.bitwiseAnd(1 << 3).neq(0)
    snow = qa.bitwiseAnd(1 << 5).neq(0)
    clear_mask = cloud.Or(cloud_shadow).Or(snow).Not()

    blue = img.select("SR_B2").multiply(0.0000275).add(-0.2).rename("blue")
    red = img.select("SR_B4").multiply(0.0000275).add(-0.2).rename("red")
    nir = img.select("SR_B5").multiply(0.0000275).add(-0.2).rename("nir")

    numerator = nir.subtract(red)
    denominator = nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    evi = numerator.divide(denominator).multiply(2.5).rename("evi")

    return evi.updateMask(clear_mask).copyProperties(img, ["system:time_start"])


def _prep_sentinel2_evi_image(img: ee.Image) -> ee.Image:
    qa = img.select("QA60")
    cloud = qa.bitwiseAnd(1 << 10).neq(0)
    cirrus = qa.bitwiseAnd(1 << 11).neq(0)
    clear_mask = cloud.Or(cirrus).Not()

    blue = img.select("B2").multiply(0.0001).rename("blue")
    red = img.select("B4").multiply(0.0001).rename("red")
    nir = img.select("B8A").multiply(0.0001).rename("nir")

    numerator = nir.subtract(red)
    denominator = nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    evi = numerator.divide(denominator).multiply(2.5).rename("evi")

    return evi.updateMask(clear_mask).copyProperties(img, ["system:time_start"])


def _get_evi_composite(region: ee.Geometry, date_start: str, date_end: str, sensor: str, cloud_pct: float) -> ee.Image:
    if sensor == "Landsat":
        col = (
            get_collection("Landsat")
            .filterDate(date_start, date_end)
            .filterBounds(region)
            .map(lambda img: _prep_landsat_evi_image(ee.Image(img)))
        )
        return col.median().rename("evi")

    elif sensor == "Sentinel2":
        col = (
            get_collection("Sentinel2")
            .filterDate(date_start, date_end)
            .filterBounds(region)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
            .map(lambda img: _prep_sentinel2_evi_image(ee.Image(img)))
        )
        return col.median().rename("evi")

    else:
        raise ValueError("EVI currently supports only Landsat or Sentinel2.")


def run_evi_method(
    region_geojson: str,
    date_start: str,
    date_end: str,
    sensor: str,
    threshold: float = 0.1,
    cloud_pct: float = 20.0,
) -> Dict[str, Any]:
    try:
        init_gee()

        region = load_region(region_geojson)
        evi_image = _get_evi_composite(region, date_start, date_end, sensor, cloud_pct)

        if sensor == "Landsat":
            base_proj = (
                get_collection("Landsat")
                .filterDate(date_start, date_end)
                .filterBounds(region)
                .first()
                .select("SR_B5")
                .projection()
            )
            native_scale = 30
        elif sensor == "Sentinel2":
            base_proj = (
                get_collection("Sentinel2")
                .filterDate(date_start, date_end)
                .filterBounds(region)
                .first()
                .select("B8A")
                .projection()
            )
            native_scale = 20
        else:
            raise ValueError("EVI currently supports only Landsat or Sentinel2.")

        evi_image = evi_image.setDefaultProjection(base_proj)
        candidate_mask = evi_image.lt(threshold).rename("candidate").setDefaultProjection(base_proj)

        evi_stats = evi_image.reduceRegion(
            reducer=ee.Reducer.min()
            .combine(ee.Reducer.max(), sharedInputs=True)
            .combine(ee.Reducer.mean(), sharedInputs=True)
            .combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=region,
            scale=native_scale,
            maxPixels=1e9,
            bestEffort=True,
        ).getInfo()

        candidate_pixels = candidate_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=native_scale,
            maxPixels=1e9,
            bestEffort=True,
        ).getInfo()

        return {
            "status": "ok",
            "method_family": "Index method",
            "method_name": "EVI",
            "sensor": sensor,
            "formula": "2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)",
            "bands": {
                "Landsat": ["SR_B2", "SR_B4", "SR_B5"],
                "Sentinel2": ["B2", "B4", "B8A"],
            },
            "date_start": date_start,
            "date_end": date_end,
            "threshold": threshold,
            "cloud_pct": cloud_pct,
            "candidate_pixel_count": int(candidate_pixels.get("candidate", 0) or 0),
            "evi_stats": {
                "min": round(float(evi_stats.get("evi_min", 0.0) or 0.0), 6),
                "max": round(float(evi_stats.get("evi_max", 0.0) or 0.0), 6),
                "mean": round(float(evi_stats.get("evi_mean", 0.0) or 0.0), 6),
                "stdDev": round(float(evi_stats.get("evi_stdDev", 0.0) or 0.0), 6),
            },
            "message": "EVI index method completed.",
        }

    except Exception as e:
        return {
            "status": "error",
            "method_family": "Index method",
            "method_name": "EVI",
            "sensor": sensor,
            "message": str(e),
        }