from __future__ import annotations

import json
import os
import re
import threading
import urllib.request
import uuid
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import shapefile
from flask import Flask, jsonify, render_template, request

from agent.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from agent.tools_gee import init_gee
from agent.wildfire_pipeline import DEFAULT_CASE, run_wildfire_pipeline

app = Flask(__name__, static_folder="static")

TASKS: dict[str, dict[str, Any]] = {}
TASK_LOCK = threading.Lock()

FLOW_STAGES = [
    {"id": "active_fire", "title": "1. Active Fire Detection"},
    {"id": "smoke_plume", "title": "2. Smoke Plume Enhancement"},
    {"id": "pre_post_change", "title": "3. Pre/Post Change Detection"},
    {"id": "burned_area", "title": "4. Burned Area Extraction"},
    {"id": "severity", "title": "5. Burn Severity Classification"},
]


@app.route("/")
def home():
    return render_template("index.html", default_case=json.dumps(DEFAULT_CASE, ensure_ascii=False))


@app.route("/authenticate_gee", methods=["POST"])
def authenticate_gee():
    try:
        init_gee()
        return jsonify({"status": "success", "message": "GEE authentication successful."})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/upload_region", methods=["POST"])
def upload_region():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"status": "error", "message": "No shapefile or zip was uploaded."}), 400

    try:
        geojson = _geojson_from_uploaded_files(files)
        return jsonify(
            {
                "status": "success",
                "region_geojson": geojson,
                "bbox": _compute_bbox(geojson),
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/start_analysis", methods=["POST"])
def start_analysis():
    payload = request.get_json() or {}
    try:
        task_config = _normalize_task_config(payload)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    return jsonify(_create_analysis_task(task_config))


@app.route("/chat_start_analysis", methods=["POST"])
def chat_start_analysis():
    payload = request.get_json() or {}
    message = (payload.get("message") or "").strip()
    language = payload.get("language", "en")

    if not message:
        return jsonify({"status": "error", "message": "Message is required."}), 400

    try:
        task_config = _build_task_config_from_chat(message, payload, language)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    task_info = _create_analysis_task(task_config)
    task_info["reply"] = _chat_start_reply(task_config, language)
    task_info["resolved_region_source"] = task_config.get("region_source", "unknown")
    task_info["resolved_task"] = {
        "case_name": task_config["case_name"],
        "date_start": task_config["date_start"],
        "date_end": task_config["date_end"],
        "region_geojson": json.loads(task_config["region_geojson"]),
        "sensors": task_config.get("sensors", ["landsat"]),
    }
    return jsonify(task_info)


@app.route("/task_status/<task_id>", methods=["GET"])
def task_status(task_id: str):
    with TASK_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return jsonify({"status": "error", "message": "Task not found."}), 404
        return jsonify(task)


def _normalize_task_config(payload: dict[str, Any]) -> dict[str, Any]:
    case_name = (payload.get("case_name") or DEFAULT_CASE["name"]).strip()
    date_start = (payload.get("date_start") or DEFAULT_CASE["date_start"]).strip()
    date_end = (payload.get("date_end") or DEFAULT_CASE["date_end"]).strip()
    region_geojson = payload.get("region_geojson")

    if not region_geojson:
        region_geojson = json.dumps(DEFAULT_CASE["region_geojson"], ensure_ascii=False)
    elif isinstance(region_geojson, dict):
        region_geojson = json.dumps(_normalize_geojson_object(region_geojson), ensure_ascii=False)
    elif isinstance(region_geojson, str):
        region_geojson = json.dumps(_normalize_geojson_object(json.loads(region_geojson)), ensure_ascii=False)
    else:
        raise ValueError("Region must be a GeoJSON string or object.")

    if not date_start or not date_end:
        raise ValueError("Start date and end date are required.")

    return {
        "case_name": case_name,
        "date_start": date_start,
        "date_end": date_end,
        "region_geojson": region_geojson,
        "region_source": payload.get("region_source", "user_custom"),
        "sensors": _normalize_sensors(payload.get("sensors")),
    }


def _create_analysis_task(task_config: dict[str, Any]) -> dict[str, Any]:
    task_id = uuid.uuid4().hex
    task_state = {
        "task_id": task_id,
        "status": "queued",
        "message": "Wildfire analysis task has been queued.",
        "case_name": task_config["case_name"],
        "sensor": ", ".join(_sensor_display_name(sensor) for sensor in task_config["sensors"]),
        "sensor_keys": task_config["sensors"],
        "date_start": task_config["date_start"],
        "date_end": task_config["date_end"],
        "region_geojson": task_config["region_geojson"],
        "flow": _initial_flow_state(),
        "stages": [],
        "downloads": {},
        "default_case": DEFAULT_CASE,
        "region_source": task_config.get("region_source", "user_custom"),
    }

    with TASK_LOCK:
        TASKS[task_id] = task_state

    worker = threading.Thread(target=_run_analysis_task, args=(task_id, task_config), daemon=True)
    worker.start()
    return {"status": "accepted", "task_id": task_id}


def _build_task_config_from_chat(message: str, payload: dict[str, Any], language: str) -> dict[str, Any]:
    custom_case_name = (payload.get("case_name") or "").strip()
    custom_date_start = (payload.get("date_start") or "").strip()
    custom_date_end = (payload.get("date_end") or "").strip()
    custom_region = payload.get("region_geojson")
    has_custom_region = bool(payload.get("has_custom_region"))
    sensors = _resolve_sensors_from_chat(message, payload.get("sensors"))

    message_dates = _extract_dates_from_message(message)
    date_start = message_dates["date_start"] or custom_date_start or DEFAULT_CASE["date_start"]
    date_end = message_dates["date_end"] or custom_date_end or DEFAULT_CASE["date_end"]

    if has_custom_region and custom_region:
        region_obj = custom_region if isinstance(custom_region, dict) else json.loads(custom_region)
        return {
            "case_name": custom_case_name or DEFAULT_CASE["name"],
            "date_start": date_start,
            "date_end": date_end,
            "region_geojson": json.dumps(_normalize_geojson_object(region_obj), ensure_ascii=False),
            "region_source": "user_custom",
            "sensors": sensors,
        }

    inferred = _infer_region_from_message(message, date_start, date_end, language)
    case_name = custom_case_name or inferred.get("case_name") or DEFAULT_CASE["name"]
    region_obj = inferred.get("region_geojson") or DEFAULT_CASE["region_geojson"]

    return {
        "case_name": case_name,
        "date_start": inferred.get("date_start", date_start),
        "date_end": inferred.get("date_end", date_end),
        "region_geojson": json.dumps(_normalize_geojson_object(region_obj), ensure_ascii=False),
        "region_source": inferred.get("region_source", "model_inferred"),
        "sensors": sensors,
    }


def _extract_dates_from_message(message: str) -> dict[str, str]:
    message = message.strip()
    match_cn_range = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})[-到至](\d{1,2})[日号]?", message)
    if match_cn_range:
        year = int(match_cn_range.group(1))
        month = int(match_cn_range.group(2))
        start_day = int(match_cn_range.group(3))
        end_day = int(match_cn_range.group(4))
        return {
            "date_start": f"{year:04d}-{month:02d}-{start_day:02d}",
            "date_end": f"{year:04d}-{month:02d}-{end_day:02d}",
        }

    match_cn_dual = re.search(
        r"(\d{4})年(\d{1,2})月(\d{1,2})[日号]?\s*[-到至]\s*(\d{4})年(\d{1,2})月(\d{1,2})[日号]?",
        message,
    )
    if match_cn_dual:
        return {
            "date_start": f"{int(match_cn_dual.group(1)):04d}-{int(match_cn_dual.group(2)):02d}-{int(match_cn_dual.group(3)):02d}",
            "date_end": f"{int(match_cn_dual.group(4)):04d}-{int(match_cn_dual.group(5)):02d}-{int(match_cn_dual.group(6)):02d}",
        }

    matches = re.findall(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", message)
    normalized = [re.sub(r"[/-]", "-", item) for item in matches]
    if len(normalized) >= 2:
        return {"date_start": _normalize_date(normalized[0]), "date_end": _normalize_date(normalized[1])}
    if len(normalized) == 1:
        date = _normalize_date(normalized[0])
        return {"date_start": date, "date_end": date}
    return {"date_start": "", "date_end": ""}


def _normalize_date(text: str) -> str:
    year, month, day = [int(item) for item in text.split("-")]
    return f"{year:04d}-{month:02d}-{day:02d}"


def _normalize_sensors(raw_sensors: Any) -> list[str]:
    if not raw_sensors:
        return ["landsat"]
    if isinstance(raw_sensors, str):
        values = [item.strip() for item in raw_sensors.split(",")]
    elif isinstance(raw_sensors, list):
        values = [str(item).strip() for item in raw_sensors]
    else:
        values = [str(raw_sensors).strip()]

    normalized: list[str] = []
    for value in values:
        key = value.lower().replace("-", "").replace("_", "")
        if key == "landsat":
            sensor_key = "landsat"
        elif key == "sentinel2":
            sensor_key = "sentinel2"
        elif key == "modis":
            sensor_key = "modis"
        elif key == "viirs":
            sensor_key = "viirs"
        else:
            continue
        if sensor_key not in normalized:
            normalized.append(sensor_key)
    return normalized or ["landsat"]


def _sensor_display_name(sensor_key: str) -> str:
    return {
        "landsat": "Landsat",
        "sentinel2": "Sentinel-2",
        "modis": "MODIS",
        "viirs": "VIIRS",
    }.get(sensor_key, sensor_key)


def _resolve_sensors_from_chat(message: str, payload_sensors: Any) -> list[str]:
    selected = _normalize_sensors(payload_sensors)
    lowered = message.lower()

    found: list[str] = []
    if "landsat" in lowered:
        found.append("landsat")
    if "sentinel" in lowered or "sentinel-2" in lowered or "sentinel2" in lowered:
        found.append("sentinel2")
    if "modis" in lowered:
        found.append("modis")
    if "viirs" in lowered:
        found.append("viirs")

    return found or selected


def _infer_region_from_message(message: str, date_start: str, date_end: str, language: str) -> dict[str, Any]:
    inferred = _call_openai_region_inference(message, date_start, date_end)
    if inferred:
        inferred["region_source"] = "model_inferred"
        return inferred

    fallback = {
        "case_name": DEFAULT_CASE["name"],
        "date_start": date_start,
        "date_end": date_end,
        "region_geojson": DEFAULT_CASE["region_geojson"],
        "region_source": "default_fallback",
    }
    return fallback


def _call_openai_region_inference(message: str, date_start: str, date_end: str) -> dict[str, Any] | None:
    if not OPENAI_API_KEY:
        return None

    base_url = (OPENAI_BASE_URL or "https://api.openai.com/v1").rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    prompt = (
        "You extract wildfire study-area defaults from a user's request. "
        "Return strict JSON only with keys: case_name, date_start, date_end, region_geojson, source_reason. "
        "region_geojson must be a Polygon or MultiPolygon around the requested wildfire area. "
        "Use a compact but reasonable bounding polygon if exact perimeter is unknown. "
        "If the request mentions Southern California wildfires in January 2025, prefer a Palisades Fire area near Pacific Palisades. "
        f"User request: {message}\n"
        f"Fallback date_start: {date_start}\n"
        f"Fallback date_end: {date_end}"
    )
    body = {
        "model": OPENAI_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=35) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        region = _normalize_geojson_object(parsed["region_geojson"])
        return {
            "case_name": parsed.get("case_name") or DEFAULT_CASE["name"],
            "date_start": parsed.get("date_start") or date_start,
            "date_end": parsed.get("date_end") or date_end,
            "region_geojson": region,
            "source_reason": parsed.get("source_reason", ""),
        }
    except Exception:
        return None


def _chat_start_reply(task_config: dict[str, Any], language: str) -> str:
    region_source = task_config.get("region_source", "unknown")
    sensors_text = ", ".join(_sensor_display_name(sensor) for sensor in task_config.get("sensors", ["landsat"]))
    if language == "zh":
        source_text = {
            "user_custom": "使用了用户自定义范围。",
            "model_inferred": "使用了模型推断范围。",
            "default_fallback": "模型未返回范围，已回退到默认案例范围。",
        }.get(region_source, "已解析研究范围。")
        return (
            f"已启动野火分析任务。\n"
            f"案例：{task_config['case_name']}\n"
            f"时间：{task_config['date_start']} 到 {task_config['date_end']}\n"
            f"数据源：{sensors_text}\n"
            f"{source_text}"
        )
    source_text = {
        "user_custom": "The uploaded/custom region was used.",
        "model_inferred": "The study region was inferred by the model.",
        "default_fallback": "Model inference was unavailable, so the default case region was used.",
    }.get(region_source, "The region has been resolved.")
    return (
        f"Wildfire analysis has started.\n"
        f"Case: {task_config['case_name']}\n"
        f"Date: {task_config['date_start']} to {task_config['date_end']}\n"
        f"Sources: {sensors_text}\n"
        f"{source_text}"
    )
    if language == "zh":
        source_text = {
            "user_custom": "使用了用户自定义范围。",
            "model_inferred": "使用了模型推断范围。",
            "default_fallback": "模型未返回范围，已回退到默认案例范围。",
        }.get(region_source, "已解析范围。")
        return (
            f"已启动野火分析任务。\n"
            f"案例：{task_config['case_name']}\n"
            f"时间：{task_config['date_start']} 至 {task_config['date_end']}\n"
            f"{source_text}"
        )

    source_text = {
        "user_custom": "The uploaded/custom region was used.",
        "model_inferred": "The study region was inferred by the model.",
        "default_fallback": "Model inference was unavailable, so the default case region was used.",
    }.get(region_source, "The region has been resolved.")
    return (
        f"Wildfire analysis has started.\n"
        f"Case: {task_config['case_name']}\n"
        f"Date: {task_config['date_start']} to {task_config['date_end']}\n"
        f"{source_text}"
    )


def _run_analysis_task(task_id: str, task_config: dict[str, Any]) -> None:
    def update_progress(stage_id: str, status: str, message: str) -> None:
        with TASK_LOCK:
            task = TASKS[task_id]
            task["status"] = "running" if status == "running" else task["status"]
            task["message"] = message
            for item in task["flow"]:
                if item["id"] == stage_id:
                    item["status"] = status
                    item["message"] = message

    try:
        with TASK_LOCK:
            TASKS[task_id]["status"] = "running"
            TASKS[task_id]["message"] = "Wildfire analysis is running."

        pipeline = run_wildfire_pipeline(
            region_geojson=task_config["region_geojson"],
            date_start=task_config["date_start"],
            date_end=task_config["date_end"],
            sensors=task_config["sensors"],
            progress_callback=update_progress,
        )

        region = pipeline["region_geojson"]
        center = _compute_center(_compute_bbox(region))
        stages_payload = []

        for stage_group in pipeline["stages"]:
            stage_payload = _build_stage_group_payload(
                task_id=task_id,
                case_name=task_config["case_name"],
                region=region,
                center=center,
                stage_group=stage_group,
                date_start=task_config["date_start"],
                date_end=task_config["date_end"],
            )
            stages_payload.append(stage_payload)

        downloads = {stage["id"]: stage["source_downloads"] for stage in stages_payload}
        algorithms = [
            {
                "id": stage["id"],
                "title": stage["title"],
                "algorithms": [
                    {
                        "source_key": item["source_key"],
                        "sensor": item["sensor"],
                        "algorithm": item["algorithm"],
                    }
                    for item in stage["source_results"]
                ],
            }
            for stage in stages_payload
        ]

        with TASK_LOCK:
            task = TASKS[task_id]
            task["status"] = "completed"
            task["message"] = "Wildfire analysis pipeline completed."
            task["stages"] = stages_payload
            task["downloads"] = downloads
            task["algorithms"] = algorithms
            task["sensor"] = pipeline["sensor"]
            task["sensor_keys"] = pipeline["sensor_keys"]
            for item in task["flow"]:
                if item["status"] == "running":
                    item["status"] = "completed"
    except Exception as exc:
        with TASK_LOCK:
            task = TASKS[task_id]
            task["status"] = "error"
            task["message"] = str(exc)
            for item in task["flow"]:
                if item["status"] == "running":
                    item["status"] = "error"
                    item["message"] = str(exc)


def _build_stage_group_payload(
    task_id: str,
    case_name: str,
    region: dict[str, Any],
    center: dict[str, float] | None,
    stage_group: dict[str, Any],
    date_start: str,
    date_end: str,
) -> dict[str, Any]:
    source_results = []
    source_downloads: dict[str, Any] = {}

    for source_output in stage_group["source_results"]:
        source_payload = _build_source_result_payload(
            task_id=task_id,
            case_name=case_name,
            region=region,
            center=center,
            stage_id=stage_group["id"],
            stage_title=stage_group["title"],
            source_output=source_output,
            date_start=date_start,
            date_end=date_end,
        )
        source_results.append(source_payload)
        source_downloads[source_payload["source_key"]] = source_payload["downloads"]

    return {
        "id": stage_group["id"],
        "title": stage_group["title"],
        "date_start": date_start,
        "date_end": date_end,
        "region": region,
        "comparison": stage_group.get("comparison", {}),
        "source_results": source_results,
        "source_downloads": source_downloads,
    }


def _build_source_result_payload(
    task_id: str,
    case_name: str,
    region: dict[str, Any],
    center: dict[str, float] | None,
    stage_id: str,
    stage_title: str,
    source_output: dict[str, Any],
    date_start: str,
    date_end: str,
) -> dict[str, Any]:
    html_stage_output = dict(source_output)
    html_stage_output["id"] = stage_id
    html_stage_output["title"] = f"{stage_title} - {source_output['sensor']}"
    html_info = _generate_stage_map(task_id, html_stage_output, region, center)
    downloads = _build_stage_downloads(
        task_id=task_id,
        case_name=case_name,
        stage_output=html_stage_output,
        region=region,
        date_start=date_start,
        date_end=date_end,
    )
    return {
        "id": stage_id,
        "source_key": source_output["source_key"],
        "sensor": source_output["sensor"],
        "title": html_stage_output["title"],
        "algorithm": source_output["algorithm"],
        "message": source_output.get("message", ""),
        "summary": source_output.get("summary", {}),
        "legend_items": source_output.get("legend_items", []),
        "date_start": date_start,
        "date_end": date_end,
        "region": region,
        "html_url": html_info.get("html_url"),
        "html_error": html_info.get("html_error"),
        "downloads": downloads,
    }


def _generate_stage_map(
    task_id: str,
    stage_output: dict[str, Any],
    region: dict[str, Any],
    center: dict[str, float] | None,
) -> dict[str, str | None]:
    try:
        ee_layer = _build_stage_tile_layer(stage_output)
        map_dir = Path(app.root_path) / "static" / "generated_maps"
        map_dir.mkdir(parents=True, exist_ok=True)
        map_name = f"{task_id}_{stage_output['id']}_{stage_output.get('source_key', 'default')}.html"
        map_path = map_dir / map_name
        _write_leaflet_ee_html(
            map_path=map_path,
            title=stage_output["title"],
            center=center or {"lat": 34.1, "lng": -118.52},
            region=region,
            ee_layer=ee_layer,
            legend_items=stage_output.get("legend_items", []),
        )
        return {
            "html_url": _static_url(f"generated_maps/{map_name}"),
            "html_error": None,
        }
    except Exception as exc:
        return {"html_url": None, "html_error": str(exc)}


def _build_stage_tile_layer(stage_output: dict[str, Any]) -> dict[str, str] | None:
    image = stage_output.get("mask_image") or stage_output.get("image")
    if image is None:
        return None
    map_id = image.getMapId(stage_output.get("mask_vis_params") or stage_output.get("vis_params") or {})
    return {
        "name": stage_output.get("layer_name", stage_output["title"]),
        "url": map_id["tile_fetcher"].url_format,
    }


def _build_stage_downloads(
    task_id: str,
    case_name: str,
    stage_output: dict[str, Any],
    region: dict[str, Any],
    date_start: str,
    date_end: str,
) -> dict[str, Any]:
    export_dir = Path(app.root_path) / "static" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    stage_stub = f"{task_id}_{stage_output['id']}_{stage_output.get('source_key', 'default')}"
    json_name = f"{stage_stub}.json"
    png_name = f"{stage_stub}.png"
    json_path = export_dir / json_name
    png_path = export_dir / png_name

    json_payload = {
        "case_name": case_name,
        "stage_id": stage_output["id"],
        "title": stage_output["title"],
        "algorithm": stage_output["algorithm"],
        "message": stage_output.get("message", ""),
        "date_start": date_start,
        "date_end": date_end,
        "sensor": stage_output.get("sensor", "Landsat"),
        "summary": stage_output.get("summary", {}),
        "legend_items": stage_output.get("legend_items", []),
        "region": region,
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_stage_png(stage_output, json_payload, png_path)

    tif_download = _build_stage_tif_download(stage_output, region, stage_stub)
    return {
        "json": {
            "label": "Download JSON",
            "url": _static_url(f"exports/{json_name}"),
            "filename": json_name,
        },
        "png": {
            "label": "Download PNG",
            "url": _static_url(f"exports/{png_name}"),
            "filename": png_name,
        },
        "tif": tif_download,
    }


def _build_stage_tif_download(stage_output: dict[str, Any], region: dict[str, Any], stage_stub: str) -> dict[str, Any]:
    try:
        image = stage_output.get("mask_image") or stage_output.get("image")
        if image is None:
            raise ValueError("No stage raster is available for TIF export.")
        geometry = region.get("geometry") if region.get("type") == "Feature" else region
        url = image.getDownloadURL(
            {
                "name": stage_stub,
                "scale": stage_output.get("download_scale", 30),
                "region": geometry,
                "fileFormat": "GeoTIFF",
            }
        )
        return {
            "label": "Download TIF",
            "url": url,
            "filename": f"{stage_stub}.tif",
            "error": None,
        }
    except Exception as exc:
        return {
            "label": "Download TIF",
            "url": None,
            "filename": f"{stage_stub}.tif",
            "error": str(exc),
        }


def _static_url(relative_path: str) -> str:
    relative_path = relative_path.replace("\\", "/").lstrip("/")
    return f"/static/{relative_path}"


def _write_stage_png(stage_output: dict[str, Any], payload: dict[str, Any], output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1180, 760), "#0b1628")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 34)
        body_font = ImageFont.truetype("arial.ttf", 22)
        small_font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        title_font = body_font = small_font = ImageFont.load_default()

    draw.text((34, 28), payload["title"], fill="#f5f8ff", font=title_font)
    draw.text((36, 78), payload["algorithm"][:110], fill="#b8c7de", font=small_font)
    draw.rounded_rectangle((34, 128, 720, 712), radius=22, fill="#132238", outline="#29476b", width=2)
    draw.text((54, 152), "Summary", fill="#f5f8ff", font=body_font)

    y = 198
    for key, value in payload.get("summary", {}).items():
        text = f"{key}: {value}"
        draw.text((54, y), text[:90], fill="#d7e3f4", font=small_font)
        y += 34
        if y > 670:
            break

    draw.text((770, 150), "Stage Metadata", fill="#f5f8ff", font=body_font)
    meta_lines = [
        f"Case: {payload['case_name']}",
        f"Sensor: {payload['sensor']}",
        f"Start: {payload['date_start']}",
        f"End: {payload['date_end']}",
        f"Message: {payload['message']}",
    ]
    meta_y = 198
    for line in meta_lines:
        draw.text((770, meta_y), line[:52], fill="#d7e3f4", font=small_font)
        meta_y += 36

    image.save(output_path, "PNG")


def _write_leaflet_ee_html(
    map_path: Path,
    title: str,
    center: dict[str, float],
    region: dict[str, Any],
    ee_layer: dict[str, str] | None,
    legend_items: list[dict[str, str]] | None,
) -> None:
    region_json = json.dumps(region, ensure_ascii=False)
    layer_json = json.dumps(ee_layer, ensure_ascii=False)
    legend_json = json.dumps(legend_items or [], ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_escape_html(title)}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body, #map {{
            width: 100%;
            height: 100%;
            min-height: 430px;
            margin: 0;
            padding: 0;
        }}
        .leaflet-control-layers {{
            font-size: 12px;
        }}
        .legend-box {{
            background: rgba(15, 23, 42, 0.92);
            color: #f8fafc;
            padding: 10px 12px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.16);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.26);
            min-width: 180px;
            line-height: 1.4;
        }}
        .legend-title {{
            margin-bottom: 8px;
            font-size: 13px;
            font-weight: 700;
        }}
        .legend-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
            font-size: 12px;
        }}
        .legend-swatch {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.28);
            flex: 0 0 16px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        const map = L.map("map").setView([{center["lat"]}, {center["lng"]}], 9);
        const osm = L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
            maxZoom: 19,
            attribution: "&copy; OpenStreetMap contributors"
        }}).addTo(map);
        const satellite = L.tileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",
            {{ attribution: "Tiles &copy; Esri" }}
        );
        const region = {region_json};
        const regionLayer = L.geoJSON(region, {{
            style: {{
                color: "#00ffff",
                weight: 2,
                fillColor: "#00ffff",
                fillOpacity: 0.10
            }}
        }}).addTo(map);
        map.fitBounds(regionLayer.getBounds(), {{ padding: [18, 18] }});

        const overlays = {{
            "Input Region": regionLayer
        }};
        const eeLayer = {layer_json};
        if (eeLayer && eeLayer.url) {{
            overlays[eeLayer.name || "Earth Engine Layer"] = L.tileLayer(eeLayer.url, {{
                opacity: 0.78,
                attribution: "Google Earth Engine"
            }}).addTo(map);
        }}

        L.control.layers({{
            "OpenStreetMap": osm,
            "Esri World Imagery": satellite
        }}, overlays, {{ collapsed: false }}).addTo(map);

        const legendItems = {legend_json};
        if (legendItems.length) {{
            const legend = L.control({{ position: "bottomright" }});
            legend.onAdd = function() {{
                const div = L.DomUtil.create("div", "legend-box");
                let html = '<div class="legend-title">Legend</div>';
                for (const item of legendItems) {{
                    const color = item.color || "#999999";
                    const label = item.label || "Layer";
                    html += `<div class="legend-row"><span class="legend-swatch" style="background:${{color}}"></span><span>${{label}}</span></div>`;
                }}
                div.innerHTML = html;
                return div;
            }};
            legend.addTo(map);
        }}
    </script>
</body>
</html>
"""
    map_path.write_text(html, encoding="utf-8")


def _geojson_from_uploaded_files(files: list[Any]) -> dict[str, Any]:
    with TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        uploaded_paths = []
        for item in files:
            filename = Path(item.filename or "").name
            if not filename:
                continue
            destination = temp_dir / filename
            item.save(destination)
            uploaded_paths.append(destination)

        zip_files = [path for path in uploaded_paths if path.suffix.lower() == ".zip"]
        if zip_files:
            with zipfile.ZipFile(zip_files[0]) as archive:
                archive.extractall(temp_dir)

        shp_files = list(temp_dir.rglob("*.shp"))
        if not shp_files:
            raise ValueError("No .shp file was found in the uploaded content.")

        reader = shapefile.Reader(str(shp_files[0]))
        shapes = reader.shapes()
        if not shapes:
            raise ValueError("The shapefile does not contain any geometry.")

        polygons: list[Any] = []
        for shp in shapes:
            geo = shp.__geo_interface__
            if geo["type"] == "Polygon":
                polygons.append(geo["coordinates"])
            elif geo["type"] == "MultiPolygon":
                polygons.extend(list(geo["coordinates"]))

        if not polygons:
            raise ValueError("Only polygon shapefiles are supported in this workflow.")

        if len(polygons) == 1:
            return {"type": "Polygon", "coordinates": polygons[0]}
        return {"type": "MultiPolygon", "coordinates": polygons}


def _normalize_geojson_object(geojson_obj: dict[str, Any]) -> dict[str, Any]:
    geo_type = geojson_obj.get("type")
    if geo_type in {"Polygon", "MultiPolygon", "Feature"}:
        return geojson_obj

    if geo_type == "FeatureCollection":
        polygons: list[Any] = []
        for feature in geojson_obj.get("features", []):
            geometry = feature.get("geometry", {})
            if geometry.get("type") == "Polygon":
                polygons.append(geometry.get("coordinates", []))
            elif geometry.get("type") == "MultiPolygon":
                polygons.extend(geometry.get("coordinates", []))
        if not polygons:
            raise ValueError("The FeatureCollection does not contain polygon geometry.")
        if len(polygons) == 1:
            return {"type": "Polygon", "coordinates": polygons[0]}
        return {"type": "MultiPolygon", "coordinates": polygons}

    raise ValueError("Only Polygon, MultiPolygon, Feature, and FeatureCollection are supported.")


def _initial_flow_state() -> list[dict[str, str]]:
    return [
        {
            "id": item["id"],
            "title": item["title"],
            "status": "pending",
            "message": "Waiting to run.",
        }
        for item in FLOW_STAGES
    ]


def _compute_bbox(region: dict[str, Any] | None) -> dict[str, float] | None:
    if not region:
        return None
    geometry = region.get("geometry") if region.get("type") == "Feature" else region
    if not isinstance(geometry, dict):
        return None
    coords = _flatten_coordinates(geometry.get("coordinates", []))
    if not coords:
        return None
    lngs = [point[0] for point in coords]
    lats = [point[1] for point in coords]
    return {
        "min_lng": min(lngs),
        "min_lat": min(lats),
        "max_lng": max(lngs),
        "max_lat": max(lats),
    }


def _compute_center(bbox: dict[str, float] | None) -> dict[str, float] | None:
    if not bbox:
        return None
    return {
        "lng": round((bbox["min_lng"] + bbox["max_lng"]) / 2, 6),
        "lat": round((bbox["min_lat"] + bbox["max_lat"]) / 2, 6),
    }


def _flatten_coordinates(coords: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if (
        isinstance(coords, list)
        and len(coords) >= 2
        and isinstance(coords[0], (int, float))
        and isinstance(coords[1], (int, float))
    ):
        return [(float(coords[0]), float(coords[1]))]
    if isinstance(coords, list):
        for item in coords:
            points.extend(_flatten_coordinates(item))
    return points


def _escape_html(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


if __name__ == "__main__":
    app.run(debug=True)
