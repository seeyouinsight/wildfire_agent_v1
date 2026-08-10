import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEE_PROJECT_ID = os.getenv("YOUR_GEE_PROJECT_ID", "")

SENSOR_COLLECTION_MAP = {
    "MODIS": {
        "collection_id": "MODIS/061/MOD14A1",
        "type": "fire_product",
        "summary_bands": ["FireMask", "MaxFRP", "QA"],
    },
    "VIIRS": {
        "collection_id": "NASA/VIIRS/002/VNP14A1",
        "type": "fire_product",
        "summary_bands": ["FireMask", "MaxFRP", "QA"],
    },
    "Landsat": {
        "collection_id": "LANDSAT/LC08/C02/T1_L2",
        "type": "reflectance_multispectral",
        "summary_bands": ["SR_B2", "SR_B4", "SR_B5", "SR_B7", "QA_PIXEL"],
    },
    "Sentinel2": {
        "collection_id": "COPERNICUS/S2_SR_HARMONIZED",
        "type": "reflectance_multispectral",
        "summary_bands": ["B2", "B4", "B8A", "B12", "QA60"],
    },
}
