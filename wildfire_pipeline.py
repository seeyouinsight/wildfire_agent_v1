from __future__ import annotations

import datetime as dt
import json
from typing import Any, Callable

import ee

from agent.utils.gee_common import load_region
from agent.utils.gee_init import init_gee


DEFAULT_CASE = {
    "name": "Palisades Fire, Southern California",
    "event_date": "2025-01-07",
    "date_start": "2025-01-07",
    "date_end": "2025-01-31",
    "source_note": "Default case centered on the CAL FIRE Palisades Fire location in Pacific Palisades.",
    "region_geojson": {
        "type": "Polygon",
        "coordinates": [[
            [-118.72, 33.98],
            [-118.31, 33.98],
            [-118.31, 34.24],
            [-118.72, 34.24],
            [-118.72, 33.98],
        ]],
    },
}

SENSOR_CONFIGS = {
    "landsat": {
        "label": "Landsat",
        "scale": 30,
        "active_fire_algorithm": (
            "Adaptive thermal anomaly on Landsat surface temperature. "
            "Threshold = regional mean + 2 * std on maximum ST_B10 composite."
        ),
        "smoke_algorithm": (
            "HOT smoke enhancement on Landsat reflectance. "
            "HOT = blue - 0.5 * red - 0.08, with adaptive threshold = mean + 0.75 * std."
        ),
        "change_algorithm": (
            "Median composite dNBR change detection from Landsat pre and post windows. "
            "dNBR = pre-fire NBR - post-fire NBR."
        ),
        "burned_area_algorithm": (
            "Adaptive burned-area extraction from Landsat dNBR. "
            "Threshold = positive dNBR mean + 0.5 * std with a fallback relaxation."
        ),
        "severity_algorithm": (
            "Standard dNBR severity bins on Landsat: unburned, low, "
            "moderate-low, moderate-high, high."
        ),
    },
    "sentinel2": {
        "label": "Sentinel-2",
        "scale": 20,
        "active_fire_algorithm": (
            "Adaptive SWIR2 anomaly on Sentinel-2 reflectance. "
            "Threshold = regional mean + 1.5 * std on B12-like SWIR2 composite, "
            "combined with SWIR dominance and moderate NIR filtering."
        ),
        "smoke_algorithm": (
            "HOT smoke enhancement on Sentinel-2 reflectance. "
            "HOT = blue - 0.5 * red - 0.08, with adaptive threshold = mean + 0.75 * std."
        ),
        "change_algorithm": (
            "Median composite dNBR change detection from Sentinel-2 pre and post windows. "
            "dNBR = pre-fire NBR - post-fire NBR."
        ),
        "burned_area_algorithm": (
            "Adaptive burned-area extraction from Sentinel-2 dNBR. "
            "Threshold = positive dNBR mean + 0.5 * std with a fallback relaxation."
        ),
        "severity_algorithm": (
            "Standard dNBR severity bins on Sentinel-2: unburned, low, "
            "moderate-low, moderate-high, high."
        ),
    },
    "modis": {
        "label": "MODIS",
        "scale": 500,
        "active_fire_algorithm": (
            "MODIS active fire mask using the MOD14A1 FireMask layer. "
            "Pixels with FireMask >= 7 are treated as fire detections."
        ),
        "smoke_algorithm": (
            "HOT smoke enhancement on MODIS surface reflectance. "
            "HOT = blue - 0.5 * red - 0.08, with adaptive threshold = mean + 0.75 * std."
        ),
        "change_algorithm": (
            "Median composite dNBR change detection from MODIS pre and post windows. "
            "dNBR = pre-fire NBR - post-fire NBR."
        ),
        "burned_area_algorithm": (
            "Adaptive burned-area extraction from MODIS dNBR. "
            "Threshold = positive dNBR mean + 0.5 * std with a fallback relaxation."
        ),
        "severity_algorithm": (
            "Standard dNBR severity bins on MODIS: unburned, low, "
            "moderate-low, moderate-high, high."
        ),
    },
    "viirs": {
        "label": "VIIRS",
        "scale": 500,
        "active_fire_algorithm": (
            "VIIRS active fire mask using the VNP14A1 FireMask layer. "
            "Pixels with FireMask >= 7 are treated as fire detections."
        ),
        "smoke_algorithm": (
            "HOT smoke enhancement on VIIRS surface reflectance. "
            "HOT = blue - 0.5 * red - 0.08, with adaptive threshold = mean + 0.75 * std."
        ),
        "change_algorithm": (
            "Median composite dNBR change detection from VIIRS pre and post windows. "
            "dNBR = pre-fire NBR - post-fire NBR."
        ),
        "burned_area_algorithm": (
            "Adaptive burned-area extraction from VIIRS dNBR. "
            "Threshold = positive dNBR mean + 0.5 * std with a fallback relaxation."
        ),
        "severity_algorithm": (
            "Standard dNBR severity bins on VIIRS: unburned, low, "
            "moderate-low, moderate-high, high."
        ),
    },
}

STAGE_DEFINITIONS = [
    {"id": "active_fire", "title": "1. Active Fire Detection"},
    {"id": "smoke_plume", "title": "2. Smoke Plume Enhancement"},
    {"id": "pre_post_change", "title": "3. Pre/Post Change Detection"},
    {"id": "burned_area", "title": "4. Burned Area Extraction"},
    {"id": "severity", "title": "5. Burn Severity Classification"},
]


def run_wildfire_pipeline(
    region_geojson: str,
    date_start: str,
    date_end: str,
    sensors: list[str] | None = None,
    progress_callback: Callable[[str, str, str], None] | None = None,
) -> dict[str, Any]:
    init_gee()
    region = load_region(region_geojson)
    sensor_keys = _normalize_sensors(sensors)

    stage_groups: list[dict[str, Any]] = []

    def notify(stage_id: str, status: str, message: str) -> None:
        if progress_callback:
            progress_callback(stage_id, status, message)

    for stage_def in STAGE_DEFINITIONS:
        stage_id = stage_def["id"]
        stage_title = stage_def["title"]
        notify(stage_id, "running", f"{stage_title} is running.")

        source_results: list[dict[str, Any]] = []
        for sensor_key in sensor_keys:
            source_results.append(_run_stage_for_sensor(stage_id, sensor_key, region, date_start, date_end))

        stage_groups.append(
            {
                "id": stage_id,
                "title": stage_title,
                "source_results": source_results,
                "comparison": _build_stage_comparison(stage_id, source_results),
            }
        )
        notify(stage_id, "completed", f"{stage_title} completed for {', '.join(_sensor_label(key) for key in sensor_keys)}.")

    return {
        "sensor": ", ".join(_sensor_label(key) for key in sensor_keys),
        "sensor_keys": sensor_keys,
        "date_start": date_start,
        "date_end": date_end,
        "region_geojson": json.loads(region_geojson),
        "stages": stage_groups,
    }


def _run_stage_for_sensor(
    stage_id: str,
    sensor_key: str,
    region: ee.Geometry,
    date_start: str,
    date_end: str,
) -> dict[str, Any]:
    if stage_id == "active_fire":
        output = detect_active_fire(region, date_start, date_end, sensor_key)
    elif stage_id == "smoke_plume":
        output = detect_smoke_plume(region, date_start, date_end, sensor_key)
    elif stage_id == "pre_post_change":
        output = detect_pre_post_change(region, date_start, date_end, sensor_key)
    elif stage_id == "burned_area":
        output = extract_burned_area(region, date_start, date_end, sensor_key)
    elif stage_id == "severity":
        output = classify_burn_severity(region, date_start, date_end, sensor_key)
    else:
        raise ValueError(f"Unsupported stage: {stage_id}")

    output["source_key"] = sensor_key
    output["sensor"] = _sensor_label(sensor_key)
    return output


def detect_active_fire(region: ee.Geometry, date_start: str, date_end: str, sensor_key: str = "landsat") -> dict[str, Any]:
    scale = _sensor_scale(sensor_key)
    if sensor_key == "modis":
        fire_mask = (
            ee.ImageCollection("MODIS/061/MOD14A1")
            .filterDate(date_start, date_end)
            .filterBounds(region)
            .select("FireMask")
            .max()
            .rename("active_fire")
            .clip(region)
        )
        hotspot_mask = fire_mask.gte(7).rename("active_fire").clip(region)
        return {
            "status": "ok",
            "message": "MODIS active fire mask detection completed.",
            "algorithm": SENSOR_CONFIGS[sensor_key]["active_fire_algorithm"],
            "summary": {
                "firemask_threshold": 7,
                "candidate_pixel_count": _pixel_sum(hotspot_mask, region, scale, "active_fire"),
            },
            "image": hotspot_mask.updateMask(hotspot_mask),
            "vis_params": {"min": 0, "max": 1, "palette": ["#ffe082", "#e53935"]},
            "layer_name": "MODIS Active Fire Mask",
            "download_scale": scale,
            "legend_items": [{"label": "MODIS fire pixel", "color": "#e53935"}],
        }

    if sensor_key == "viirs":
        fire_mask = (
            ee.ImageCollection("NASA/VIIRS/002/VNP14A1")
            .filterDate(date_start, date_end)
            .filterBounds(region)
            .select("FireMask")
            .max()
            .rename("active_fire")
            .clip(region)
        )
        hotspot_mask = fire_mask.gte(7).rename("active_fire").clip(region)
        return {
            "status": "ok",
            "message": "VIIRS active fire mask detection completed.",
            "algorithm": SENSOR_CONFIGS[sensor_key]["active_fire_algorithm"],
            "summary": {
                "firemask_threshold": 7,
                "candidate_pixel_count": _pixel_sum(hotspot_mask, region, scale, "active_fire"),
            },
            "image": hotspot_mask.updateMask(hotspot_mask),
            "vis_params": {"min": 0, "max": 1, "palette": ["#fff176", "#fb8c00", "#d32f2f"]},
            "layer_name": "VIIRS Active Fire Mask",
            "download_scale": scale,
            "legend_items": [{"label": "VIIRS fire pixel", "color": "#d32f2f"}],
        }

    collection = _sensor_collection(sensor_key, region, date_start, date_end)

    if sensor_key == "landsat":
        composite = collection.select("temp").max().rename("temp").clip(region)
        stats = _stats_for_band(composite, region, "temp", scale)
        threshold = stats["mean"] + 2.0 * stats["stdDev"]
        hotspot_mask = composite.gt(threshold).rename("active_fire").clip(region)
        summary = {
            "adaptive_threshold_kelvin": round(threshold, 4),
            "temperature_stats": stats,
            "candidate_pixel_count": _pixel_sum(hotspot_mask, region, scale, "active_fire"),
        }
        return {
            "status": "ok",
            "message": "Adaptive active fire detection completed.",
            "algorithm": SENSOR_CONFIGS[sensor_key]["active_fire_algorithm"],
            "summary": summary,
            "image": hotspot_mask.updateMask(hotspot_mask),
            "vis_params": {"min": 0, "max": 1, "palette": ["#ffee58", "#ff9800", "#d32f2f"]},
            "layer_name": "Adaptive Active Fire",
            "download_scale": scale,
            "legend_items": [{"label": "Potential active fire pixel", "color": "#d32f2f"}],
        }

    composite = collection.select("swir2").max().rename("swir2").clip(region)
    stats = _stats_for_band(composite, region, "swir2", scale)
    threshold = stats["mean"] + 1.5 * stats["stdDev"]
    median_comp = collection.median().clip(region)
    hotspot_mask = (
        composite.gt(threshold)
        .And(median_comp.select("swir2").gt(median_comp.select("swir1")))
        .And(median_comp.select("nir").lt(0.6))
        .rename("active_fire")
        .clip(region)
    )
    summary = {
        "adaptive_threshold_swir2": round(threshold, 6),
        "swir2_stats": stats,
        "candidate_pixel_count": _pixel_sum(hotspot_mask, region, scale, "active_fire"),
    }
    return {
        "status": "ok",
        "message": "Adaptive active fire detection completed.",
        "algorithm": SENSOR_CONFIGS[sensor_key]["active_fire_algorithm"],
        "summary": summary,
        "image": hotspot_mask.updateMask(hotspot_mask),
        "vis_params": {"min": 0, "max": 1, "palette": ["#fff176", "#ffb300", "#e65100"]},
        "layer_name": "Adaptive Active Fire",
        "download_scale": scale,
        "legend_items": [{"label": "Potential active fire pixel", "color": "#e65100"}],
    }


def detect_smoke_plume(region: ee.Geometry, date_start: str, date_end: str, sensor_key: str = "landsat") -> dict[str, Any]:
    scale = _sensor_scale(sensor_key)
    composite = _sensor_collection(sensor_key, region, date_start, date_end).median().clip(region)
    hot = (
        composite.select("blue")
        .subtract(composite.select("red").multiply(0.5))
        .subtract(0.08)
        .rename("hot")
        .clip(region)
    )
    stats = _stats_for_band(hot, region, "hot", scale)
    threshold = stats["mean"] + 0.75 * stats["stdDev"]
    smoke_mask = hot.gt(threshold).And(composite.select("nir").lt(0.35)).rename("smoke").clip(region)

    return {
        "status": "ok",
        "message": "Smoke plume enhancement completed.",
        "algorithm": SENSOR_CONFIGS[sensor_key]["smoke_algorithm"],
        "summary": {
            "adaptive_threshold": round(threshold, 6),
            "hot_stats": stats,
            "candidate_pixel_count": _pixel_sum(smoke_mask, region, scale, "smoke"),
        },
        "image": hot,
        "mask_image": smoke_mask.updateMask(smoke_mask),
        "vis_params": {"min": -0.2, "max": 0.2, "palette": ["#1b4f72", "#f7f7f7", "#8e5a2b"]},
        "mask_vis_params": {"min": 0, "max": 1, "palette": ["#d7ccc8", "#757575"]},
        "layer_name": "Smoke HOT Index",
        "download_scale": scale,
        "legend_items": [
            {"label": "Lower HOT", "color": "#1b4f72"},
            {"label": "Higher HOT", "color": "#8e5a2b"},
            {"label": "Smoke candidate mask", "color": "#757575"},
        ],
    }


def detect_pre_post_change(region: ee.Geometry, date_start: str, date_end: str, sensor_key: str = "landsat") -> dict[str, Any]:
    scale = _sensor_scale(sensor_key)
    pre_start, pre_end, post_start, post_end = _pre_post_windows(date_start, date_end)
    pre_nbr = _nbr_image(sensor_key, region, pre_start, pre_end)
    post_nbr = _nbr_image(sensor_key, region, post_start, post_end)
    d_nbr = pre_nbr.subtract(post_nbr).rename("dnbr").clip(region)
    stats = _stats_for_band(d_nbr, region, "dnbr", scale)

    return {
        "status": "ok",
        "message": "Pre/post change detection completed.",
        "algorithm": SENSOR_CONFIGS[sensor_key]["change_algorithm"],
        "summary": {
            "pre_window": {"start": pre_start, "end": pre_end},
            "post_window": {"start": post_start, "end": post_end},
            "dnbr_stats": stats,
        },
        "image": d_nbr,
        "vis_params": {"min": -0.2, "max": 0.8, "palette": ["#2166ac", "#f7f7f7", "#b2182b"]},
        "layer_name": "dNBR Change",
        "download_scale": scale,
        "legend_items": [
            {"label": "Negative / low change", "color": "#2166ac"},
            {"label": "Near zero change", "color": "#f7f7f7"},
            {"label": "Positive burn-related change", "color": "#b2182b"},
        ],
    }


def extract_burned_area(region: ee.Geometry, date_start: str, date_end: str, sensor_key: str = "landsat") -> dict[str, Any]:
    scale = _sensor_scale(sensor_key)
    change = detect_pre_post_change(region, date_start, date_end, sensor_key)
    d_nbr = change["image"]
    positive = d_nbr.updateMask(d_nbr.gt(0))
    stats = _stats_for_band(positive, region, "dnbr", scale)
    threshold = stats["mean"] + 0.5 * stats["stdDev"]
    fallback_used = False
    if threshold <= 0:
        threshold = 0.1
        fallback_used = True

    burn_mask = d_nbr.gt(threshold).rename("burned_area").clip(region)
    candidate_pixels = _pixel_sum(burn_mask, region, scale, "burned_area")
    if candidate_pixels == 0:
        threshold = max(0.1, stats["mean"] + 0.25 * stats["stdDev"])
        burn_mask = d_nbr.gt(threshold).rename("burned_area").clip(region)
        candidate_pixels = _pixel_sum(burn_mask, region, scale, "burned_area")
        fallback_used = True

    return {
        "status": "ok",
        "message": "Burned area extraction completed.",
        "algorithm": SENSOR_CONFIGS[sensor_key]["burned_area_algorithm"],
        "summary": {
            "adaptive_threshold": round(threshold, 6),
            "positive_dnbr_stats": stats,
            "candidate_pixel_count": candidate_pixels,
            "fallback_threshold_used": fallback_used,
        },
        "image": burn_mask.updateMask(burn_mask),
        "vis_params": {"min": 0, "max": 1, "palette": ["#ffe082", "#e65100"]},
        "layer_name": "Burned Area Mask",
        "download_scale": scale,
        "legend_items": [{"label": "Burned area candidate", "color": "#e65100"}],
    }


def classify_burn_severity(region: ee.Geometry, date_start: str, date_end: str, sensor_key: str = "landsat") -> dict[str, Any]:
    scale = _sensor_scale(sensor_key)
    change = detect_pre_post_change(region, date_start, date_end, sensor_key)
    d_nbr = change["image"]

    severity = (
        ee.Image(0)
        .where(d_nbr.gte(0.1).And(d_nbr.lt(0.27)), 1)
        .where(d_nbr.gte(0.27).And(d_nbr.lt(0.44)), 2)
        .where(d_nbr.gte(0.44).And(d_nbr.lt(0.66)), 3)
        .where(d_nbr.gte(0.66), 4)
        .rename("severity")
        .clip(region)
    )
    class_stats = severity.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=region,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()
    histogram = class_stats.get("severity", {}) if isinstance(class_stats, dict) else {}

    return {
        "status": "ok",
        "message": "Burn severity classification completed.",
        "algorithm": SENSOR_CONFIGS[sensor_key]["severity_algorithm"],
        "summary": {
            "class_histogram": histogram,
            "class_labels": {
                "0": "Unburned / background",
                "1": "Low severity",
                "2": "Moderate-low severity",
                "3": "Moderate-high severity",
                "4": "High severity",
            },
        },
        "image": severity,
        "vis_params": {
            "min": 0,
            "max": 4,
            "palette": ["#4caf50", "#ffeb3b", "#ff9800", "#ef6c00", "#b71c1c"],
        },
        "layer_name": "Burn Severity",
        "download_scale": scale,
        "legend_items": [
            {"label": "Unburned / background", "color": "#4caf50"},
            {"label": "Low severity", "color": "#ffeb3b"},
            {"label": "Moderate-low severity", "color": "#ff9800"},
            {"label": "Moderate-high severity", "color": "#ef6c00"},
            {"label": "High severity", "color": "#b71c1c"},
        ],
    }


def _normalize_sensors(sensors: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for sensor in sensors or ["landsat"]:
        key = str(sensor).strip().lower().replace("-", "").replace("_", "")
        if key == "sentinel2":
            key = "sentinel2"
        elif key == "landsat":
            key = "landsat"
        elif key == "modis":
            key = "modis"
        elif key == "viirs":
            key = "viirs"
        else:
            continue
        if key not in normalized:
            normalized.append(key)
    return normalized or ["landsat"]


def _sensor_label(sensor_key: str) -> str:
    return SENSOR_CONFIGS[sensor_key]["label"]


def _sensor_scale(sensor_key: str) -> int:
    return int(SENSOR_CONFIGS[sensor_key]["scale"])


def _sensor_collection(sensor_key: str, region: ee.Geometry, date_start: str, date_end: str) -> ee.ImageCollection:
    if sensor_key == "landsat":
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        return (
            l8.merge(l9)
            .filterDate(date_start, date_end)
            .filterBounds(region)
            .map(_prep_landsat_image)
        )

    if sensor_key == "sentinel2":
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        return s2.filterDate(date_start, date_end).filterBounds(region).map(_prep_sentinel2_image)

    if sensor_key == "modis":
        modis = ee.ImageCollection("MODIS/061/MOD09GA")
        return modis.filterDate(date_start, date_end).filterBounds(region).map(_prep_modis_image)

    viirs = ee.ImageCollection("NASA/VIIRS/002/VNP09GA")
    return viirs.filterDate(date_start, date_end).filterBounds(region).map(_prep_viirs_image)


def _prep_landsat_image(img: ee.Image) -> ee.Image:
    qa = img.select("QA_PIXEL")
    cloud_shadow = qa.bitwiseAnd(1 << 4).neq(0)
    cloud = qa.bitwiseAnd(1 << 3).neq(0)
    snow = qa.bitwiseAnd(1 << 5).neq(0)
    clear_mask = cloud.Or(cloud_shadow).Or(snow).Not()

    blue = img.select("SR_B2").multiply(0.0000275).add(-0.2).rename("blue")
    red = img.select("SR_B4").multiply(0.0000275).add(-0.2).rename("red")
    nir = img.select("SR_B5").multiply(0.0000275).add(-0.2).rename("nir")
    swir1 = img.select("SR_B6").multiply(0.0000275).add(-0.2).rename("swir1")
    swir2 = img.select("SR_B7").multiply(0.0000275).add(-0.2).rename("swir2")
    temp = img.select("ST_B10").multiply(0.00341802).add(149.0).rename("temp")

    return ee.Image.cat([blue, red, nir, swir1, swir2, temp]).updateMask(clear_mask).copyProperties(
        img, ["system:time_start"]
    )


def _prep_sentinel2_image(img: ee.Image) -> ee.Image:
    scl = img.select("SCL")
    clear_mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    blue = img.select("B2").multiply(0.0001).rename("blue")
    red = img.select("B4").multiply(0.0001).rename("red")
    nir = img.select("B8").multiply(0.0001).rename("nir")
    swir1 = img.select("B11").multiply(0.0001).rename("swir1")
    swir2 = img.select("B12").multiply(0.0001).rename("swir2")

    return ee.Image.cat([blue, red, nir, swir1, swir2]).updateMask(clear_mask).copyProperties(
        img, ["system:time_start"]
    )


def _prep_modis_image(img: ee.Image) -> ee.Image:
    blue = img.select("sur_refl_b03").multiply(0.0001).rename("blue")
    red = img.select("sur_refl_b01").multiply(0.0001).rename("red")
    nir = img.select("sur_refl_b02").multiply(0.0001).rename("nir")
    swir1 = img.select("sur_refl_b06").multiply(0.0001).rename("swir1")
    swir2 = img.select("sur_refl_b07").multiply(0.0001).rename("swir2")
    valid_mask = blue.gt(0).And(red.gt(0)).And(nir.gt(0))

    return ee.Image.cat([blue, red, nir, swir1, swir2]).updateMask(valid_mask).copyProperties(
        img, ["system:time_start"]
    )


def _prep_viirs_image(img: ee.Image) -> ee.Image:
    blue = img.select("M3").multiply(0.0001).rename("blue")
    red = img.select("M5").multiply(0.0001).rename("red")
    nir = img.select("M7").multiply(0.0001).rename("nir")
    swir1 = img.select("M10").multiply(0.0001).rename("swir1")
    swir2 = img.select("M11").multiply(0.0001).rename("swir2")
    valid_mask = blue.gt(0).And(red.gt(0)).And(nir.gt(0))

    return ee.Image.cat([blue, red, nir, swir1, swir2]).updateMask(valid_mask).copyProperties(
        img, ["system:time_start"]
    )


def _nbr_image(sensor_key: str, region: ee.Geometry, date_start: str, date_end: str) -> ee.Image:
    composite = _sensor_collection(sensor_key, region, date_start, date_end).median().clip(region)
    return composite.normalizedDifference(["nir", "swir2"]).rename("dnbr").clip(region)


def _stats_for_band(image: ee.Image, region: ee.Geometry, band: str, scale: int) -> dict[str, float]:
    stats = image.reduceRegion(
        reducer=ee.Reducer.min()
        .combine(ee.Reducer.max(), sharedInputs=True)
        .combine(ee.Reducer.mean(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=region,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()
    stats = stats or {}
    return {
        "min": round(float(stats.get(f"{band}_min", 0.0) or 0.0), 6),
        "max": round(float(stats.get(f"{band}_max", 0.0) or 0.0), 6),
        "mean": round(float(stats.get(f"{band}_mean", 0.0) or 0.0), 6),
        "stdDev": round(float(stats.get(f"{band}_stdDev", 0.0) or 0.0), 6),
    }


def _pixel_sum(mask: ee.Image, region: ee.Geometry, scale: int, band: str) -> int:
    result = mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()
    result = result or {}
    return int(result.get(band, 0) or 0)


def _pre_post_windows(date_start: str, date_end: str) -> tuple[str, str, str, str]:
    start_dt = dt.date.fromisoformat(date_start)
    end_dt = dt.date.fromisoformat(date_end)
    span_days = max((end_dt - start_dt).days + 1, 16)

    pre_end = start_dt - dt.timedelta(days=1)
    pre_start = pre_end - dt.timedelta(days=span_days - 1)
    post_start = end_dt + dt.timedelta(days=1)
    post_end = post_start + dt.timedelta(days=span_days - 1)

    return (
        pre_start.isoformat(),
        pre_end.isoformat(),
        post_start.isoformat(),
        post_end.isoformat(),
    )


def _build_stage_comparison(stage_id: str, source_results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []

    for result in source_results:
        metrics.append(
            {
                "source_key": result["source_key"],
                "sensor": result["sensor"],
                "primary_value": _stage_primary_value(stage_id, result.get("summary", {})),
            }
        )

    return {
        "metric_label": _comparison_metric_label(stage_id),
        "metrics": metrics,
    }


def _comparison_metric_label(stage_id: str) -> str:
    mapping = {
        "active_fire": "Candidate Pixels",
        "smoke_plume": "Candidate Pixels",
        "pre_post_change": "dNBR Mean",
        "burned_area": "Candidate Pixels",
        "severity": "High Severity Pixels",
    }
    return mapping.get(stage_id, "Primary Metric")


def _stage_primary_value(stage_id: str, summary: dict[str, Any]) -> Any:
    if stage_id in {"active_fire", "smoke_plume", "burned_area"}:
        return summary.get("candidate_pixel_count", 0)
    if stage_id == "pre_post_change":
        return (summary.get("dnbr_stats") or {}).get("mean", 0)
    if stage_id == "severity":
        histogram = summary.get("class_histogram") or {}
        return int(histogram.get("4", 0) or 0)
    return None
