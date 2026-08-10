import json
import ee
from typing import Dict, Any

from agent.config import SENSOR_COLLECTION_MAP
from agent.utils.gee_init import init_gee
from agent.utils.gee_common import load_region, get_collection
from agent.utils.metrics import compute_metrics


def _prep_landsat_nbr_image(img: ee.Image) -> ee.Image:
    qa = img.select("QA_PIXEL")
    cloud_shadow = qa.bitwiseAnd(1 << 4).neq(0)
    cloud = qa.bitwiseAnd(1 << 3).neq(0)
    snow = qa.bitwiseAnd(1 << 5).neq(0)
    clear_mask = cloud.Or(cloud_shadow).Or(snow).Not()

    nir = img.select("SR_B5").multiply(0.0000275).add(-0.2).rename("nir")
    swir2 = img.select("SR_B7").multiply(0.0000275).add(-0.2).rename("swir2")

    return ee.Image.cat([nir, swir2]).updateMask(clear_mask).copyProperties(img, ["system:time_start"])


def _prep_sentinel2_nbr_image(img: ee.Image) -> ee.Image:
    qa = img.select("QA60")
    cloud = qa.bitwiseAnd(1 << 10).neq(0)
    cirrus = qa.bitwiseAnd(1 << 11).neq(0)
    clear_mask = cloud.Or(cirrus).Not()

    nir = img.select("B8A").multiply(0.0001).rename("nir")
    swir2 = img.select("B12").multiply(0.0001).rename("swir2")

    return ee.Image.cat([nir, swir2]).updateMask(clear_mask).copyProperties(img, ["system:time_start"])


def _get_nbr_composite(region: ee.Geometry, date_start: str, date_end: str, sensor: str, cloud_pct: float) -> ee.Image:
    if sensor == "Landsat":
        col = (
            get_collection("Landsat")
            .filterDate(date_start, date_end)
            .filterBounds(region)
            .map(_prep_landsat_nbr_image)
        )
        return col.median().normalizedDifference(["nir", "swir2"]).rename("nbr")

    elif sensor == "Sentinel2":
        col = (
            get_collection("Sentinel2")
            .filterDate(date_start, date_end)
            .filterBounds(region)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
            .map(_prep_sentinel2_nbr_image)
        )
        return col.median().normalizedDifference(["nir", "swir2"]).rename("nbr")

    else:
        raise ValueError("NBR currently supports only Landsat or Sentinel2.")


def _get_reference_fire_mask(region: ee.Geometry, date_start: str, date_end: str, reference_sensor: str) -> ee.Image:
    ref_col = (
        get_collection(reference_sensor)
        .filterDate(date_start, date_end)
        .filterBounds(region)
        .map(lambda img: ee.Image(img).select("FireMask").gt(0).rename("ref_fire"))
    )
    return ee.ImageCollection(ref_col).max().rename("ref_fire")


def run_nbr_index_method(
    region_geojson: str,
    date_start: str,
    date_end: str,
    sensor: str,
    threshold: float = 0.1,
    cloud_pct: float = 20.0,
    reference_sensor: str = "VIIRS",
) -> Dict[str, Any]:
    try:
        init_gee()

        region = load_region(region_geojson)
        nbr = _get_nbr_composite(region, date_start, date_end, sensor, cloud_pct)

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
            raise ValueError("NBR currently supports only Landsat or Sentinel2.")

        nbr = nbr.setDefaultProjection(base_proj)
        candidate_mask = nbr.lt(threshold).rename("candidate").setDefaultProjection(base_proj)

        reference_mask = _get_reference_fire_mask(region, date_start, date_end, reference_sensor)
        ref_proj = reference_mask.projection()

        candidate_1km = (
            candidate_mask
            .reduceResolution(reducer=ee.Reducer.max(), maxPixels=4096)
            .reproject(crs=ref_proj, scale=1000)
            .rename("candidate")
        )
        reference_mask = reference_mask.reproject(crs=ref_proj, scale=1000).rename("ref_fire")

        tp_img = candidate_1km.And(reference_mask).rename("tp")
        fp_img = candidate_1km.And(reference_mask.Not()).rename("fp")
        fn_img = candidate_1km.Not().And(reference_mask).rename("fn")
        tn_img = candidate_1km.Not().And(reference_mask.Not()).rename("tn")

        metrics_dict = ee.Image.cat([tp_img, fp_img, fn_img, tn_img]).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=1000,
            maxPixels=1e9,
            bestEffort=True,
        ).getInfo()

        tp = int(metrics_dict.get("tp", 0) or 0)
        fp = int(metrics_dict.get("fp", 0) or 0)
        fn = int(metrics_dict.get("fn", 0) or 0)
        tn = int(metrics_dict.get("tn", 0) or 0)

        metrics = compute_metrics(tp, fp, fn, tn)

        nbr_stats = nbr.reduceRegion(
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
            "method_name": "NBR",
            "sensor": sensor,
            "formula": "(NIR - SWIR2) / (NIR + SWIR2)",
            "bands": {
                "Landsat": ["SR_B5", "SR_B7"],
                "Sentinel2": ["B8A", "B12"],
            },
            "date_start": date_start,
            "date_end": date_end,
            "threshold": threshold,
            "cloud_pct": cloud_pct,
            "reference_sensor": reference_sensor,
            "candidate_pixel_count": int(candidate_pixels.get("candidate", 0) or 0),
            "nbr_stats": {
                "min": round(float(nbr_stats.get("nbr_min", 0.0) or 0.0), 6),
                "max": round(float(nbr_stats.get("nbr_max", 0.0) or 0.0), 6),
                "mean": round(float(nbr_stats.get("nbr_mean", 0.0) or 0.0), 6),
                "stdDev": round(float(nbr_stats.get("nbr_stdDev", 0.0) or 0.0), 6),
            },
            "evaluation": metrics,
            "message": "NBR index method completed with proxy evaluation.",
        }

    except Exception as e:
        return {
            "status": "error",
            "method_family": "Index method",
            "method_name": "NBR",
            "sensor": sensor,
            "message": str(e),
        }