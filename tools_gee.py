import os
import json
from typing import List

import ee
from pydantic import BaseModel, Field
from langchain.tools import tool
from agent.methods.method_viirs_quick import run_quick_fire_detection as _run_quick_fire_detection
from agent.methods.method_nbr import run_nbr_index_method as _run_nbr_index_method
from agent.methods.method_evi import run_evi_method as _run_evi_method

# =====================================
# GEE 初始化
# =====================================
_GEE_INITIALIZED = False

def init_gee() -> None:
    global _GEE_INITIALIZED
    if _GEE_INITIALIZED:
        return
    proxy = os.getenv("HTTP_PROXY", "").strip()
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = os.getenv("HTTPS_PROXY", proxy)

    project_id = os.getenv("YOUR_GEE_PROJECT_ID", "").strip()
    if not project_id:
        raise RuntimeError("Please set YOUR_GEE_PROJECT_ID before initializing Google Earth Engine.")

    try:
        ee.Initialize(project=project_id)
        _GEE_INITIALIZED = True
        print(f"GEE initialized with project: {project_id}")
        return
    except Exception:
        pass

    ee.Authenticate(auth_mode="localhost")
    ee.Initialize(project=project_id)
    _GEE_INITIALIZED = True


# =====================================
# 数据源配置
# =====================================
SENSOR_COLLECTION_MAP = {
    "MODIS": {
        "collection_id": "MODIS/061/MOD14A1",
        "type": "fire_product",
        "summary_bands": ["FireMask", "MaxFRP", "QA"]
    },
    "VIIRS": {
        "collection_id": "NASA/VIIRS/002/VNP14A1",
        "type": "fire_product",
        "summary_bands": ["FireMask", "MaxFRP", "QA"]
    },
    "Landsat": {
        "collection_id": "LANDSAT/LC08/C02/T1_L2",
        "type": "reflectance_thermal",
        "summary_bands": ["ST_B10"]
    },
    "Sentinel2": {
        "collection_id": "COPERNICUS/S2_SR_HARMONIZED",
        "type": "reflectance_multispectral",
        "summary_bands": ["B2", "B4", "B8A", "B12", "QA60"]
    }
}


# =====================================
# 输入 Schema
# =====================================
class GeeSearchInput(BaseModel):
    region_geojson: str = Field(description="Region geometry in GeoJSON format")
    date_start: str = Field(description="Start date in YYYY-MM-DD")
    date_end: str = Field(description="End date in YYYY-MM-DD")
    sensor: str = Field(description="Sensor name, e.g. MODIS, VIIRS, Landsat, Sentinel2")


class QuickFireInput(BaseModel):
    region_geojson: str = Field(description="Region geometry in GeoJSON format")
    date_start: str = Field(description="Start date in YYYY-MM-DD")
    date_end: str = Field(description="End date in YYYY-MM-DD")
    sensor: str = Field(default="VIIRS", description="Sensor name, usually MODIS or VIIRS")


class NBRMethodInput(BaseModel):
    region_geojson: str = Field(description="Region geometry in GeoJSON format")
    date_start: str = Field(description="Start date in YYYY-MM-DD")
    date_end: str = Field(description="End date in YYYY-MM-DD")
    sensor: str = Field(description="Sensor name: Landsat or Sentinel2")
    threshold: float = Field(default=0.1, description="NBR threshold for anomaly extraction")
    cloud_pct: float = Field(default=20.0, description="Max cloud percentage for image filtering")
    reference_sensor: str = Field(default="VIIRS", description="Reference fire product sensor")


class EVIMethodInput(BaseModel):
    region_geojson: str = Field(description="Region geometry in GeoJSON format")
    date_start: str = Field(description="Start date in YYYY-MM-DD")
    date_end: str = Field(description="End date in YYYY-MM-DD")
    sensor: str = Field(description="Sensor name: Landsat or Sentinel2")
    threshold: float = Field(default=0.1, description="EVI threshold for anomaly extraction")
    cloud_pct: float = Field(default=20.0, description="Max cloud percentage for image filtering")


# =====================================
# 工具函数
# =====================================
def _load_region(region_geojson: str) -> ee.Geometry:
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


def _get_collection(sensor: str) -> ee.ImageCollection:
    sensor = sensor.strip()
    if sensor not in SENSOR_COLLECTION_MAP:
        raise ValueError(
            f"Unsupported sensor: {sensor}. Supported sensors: {list(SENSOR_COLLECTION_MAP.keys())}"
        )
    collection_id = SENSOR_COLLECTION_MAP[sensor]["collection_id"]
    return ee.ImageCollection(collection_id)


def _safe_band_names(img: ee.Image) -> List[str]:
    try:
        return img.bandNames().getInfo() or []
    except Exception:
        return []


def _safe_count(col: ee.ImageCollection) -> int:
    try:
        return int(col.size().getInfo())
    except Exception:
        return 0


# =====================================
# LangChain tools
# =====================================
@tool(args_schema=GeeSearchInput)
def search_gee_dataset(
    region_geojson: str,
    date_start: str,
    date_end: str,
    sensor: str
) -> str:
    """Search an Earth Engine dataset by region, date range, and sensor."""
    try:
        init_gee()
        region = _load_region(region_geojson)
        collection = _get_collection(sensor).filterDate(date_start, date_end).filterBounds(region)

        count = _safe_count(collection)
        first_img = ee.Image(collection.first()) if count > 0 else None
        band_names = _safe_band_names(first_img) if first_img else []

        result = {
            "status": "ok",
            "sensor": sensor,
            "collection_id": SENSOR_COLLECTION_MAP[sensor]["collection_id"],
            "date_start": date_start,
            "date_end": date_end,
            "image_count": count,
            "bands": band_names,
            "message": f"Found {count} images in the specified region and time range."
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "sensor": sensor,
            "message": str(e)
        }, ensure_ascii=False)


@tool(args_schema=QuickFireInput)
def run_quick_fire_detection_tool(
    region_geojson: str,
    date_start: str,
    date_end: str,
    sensor: str = "VIIRS"
) -> str:
    """Run a quick regional active fire detection workflow using MODIS or VIIRS fire products."""
    init_gee()
    result = _run_quick_fire_detection(
        region_geojson=region_geojson,
        date_start=date_start,
        date_end=date_end,
        sensor=sensor
    )
    return json.dumps(result, ensure_ascii=False)


@tool(args_schema=NBRMethodInput)
def run_nbr_index_method_tool(
    region_geojson: str,
    date_start: str,
    date_end: str,
    sensor: str,
    threshold: float = 0.1,
    cloud_pct: float = 20.0,
    reference_sensor: str = "VIIRS"
) -> str:
    """Run NBR index method for fire/burn-sensitive anomaly extraction."""
    init_gee()
    result = _run_nbr_index_method(
        region_geojson=region_geojson,
        date_start=date_start,
        date_end=date_end,
        sensor=sensor,
        threshold=threshold,
        cloud_pct=cloud_pct,
        reference_sensor=reference_sensor
    )
    return json.dumps(result, ensure_ascii=False)


@tool(args_schema=EVIMethodInput)
def run_evi_method_tool(
    region_geojson: str,
    date_start: str,
    date_end: str,
    sensor: str,
    threshold: float = 0.1,
    cloud_pct: float = 20.0
) -> str:
    """Run EVI index method for anomaly extraction."""
    init_gee()
    result = _run_evi_method(
        region_geojson=region_geojson,
        date_start=date_start,
        date_end=date_end,
        sensor=sensor,
        threshold=threshold,
        cloud_pct=cloud_pct
    )
    return json.dumps(result, ensure_ascii=False)
