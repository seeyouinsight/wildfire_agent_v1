import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from agent.tools_gee import (
    init_gee,
    run_evi_method_tool,
    run_nbr_index_method_tool,
    run_quick_fire_detection_tool,
)


SUPPORTED_LANGUAGES = {"en", "zh"}
SUPPORTED_SENSORS = {"VIIRS", "MODIS", "Landsat", "Sentinel2"}
SUPPORTED_METHODS = {"quick_fire_detection", "NBR", "EVI"}


@dataclass
class ConversationState:
    gee_authenticated: bool = False
    satellite: Optional[str] = None
    method: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    region_geojson: Optional[str] = None
    last_result: Optional[Dict[str, Any]] = None


class FireDetectionChatController:
    def __init__(self, language: str = "en") -> None:
        self.language = language if language in SUPPORTED_LANGUAGES else "en"

    def handle_message(self, message: str, state_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state = self._load_state(state_dict)
        message = (message or "").strip()

        if not message:
            return self._response(
                state=state,
                reply=self._text(
                    "Please enter a task, for example: authenticate GEE, set VIIRS quick fire detection from 2024-08-01 to 2024-08-31, then run it.",
                    "请输入任务，例如：验证 GEE，设置 VIIRS 快速火点识别，时间为 2024-08-01 到 2024-08-31，然后执行。",
                ),
            )

        if self._is_reset(message):
            state = ConversationState()
            return self._response(
                state=state,
                reply=self._text(
                    "The conversation state has been reset. You can now provide a new fire detection task.",
                    "会话状态已重置，你现在可以重新输入新的火点识别任务。",
                ),
            )

        updated_fields = self._extract_and_update_state(message, state)
        intents = self._detect_intents(message)
        replies = []

        if updated_fields:
            replies.append(self._format_update_message(updated_fields, state))

        if intents["authenticate"]:
            auth_result = self._authenticate(state)
            replies.append(auth_result)

        if intents["status"]:
            replies.append(self._format_status(state))

        should_run = intents["run"] or self._looks_like_full_task(message, state)
        if should_run:
            run_result = self._run_detection(state)
            replies.append(run_result["reply"])
            if run_result["ok"]:
                state.last_result = run_result["result"]

        if not replies:
            replies.append(self._guidance_message(state))

        return self._response(state=state, reply="\n\n".join(replies))

    def _load_state(self, state_dict: Optional[Dict[str, Any]]) -> ConversationState:
        if not state_dict:
            return ConversationState()

        return ConversationState(
            gee_authenticated=bool(state_dict.get("gee_authenticated", False)),
            satellite=state_dict.get("satellite"),
            method=state_dict.get("method"),
            date_start=state_dict.get("date_start"),
            date_end=state_dict.get("date_end"),
            region_geojson=state_dict.get("region_geojson"),
            last_result=state_dict.get("last_result"),
        )

    def _response(self, state: ConversationState, reply: str) -> Dict[str, Any]:
        return {
            "reply": reply,
            "state": asdict(state),
            "result": state.last_result,
        }

    def _text(self, en: str, zh: str) -> str:
        return zh if self.language == "zh" else en

    def _detect_intents(self, message: str) -> Dict[str, bool]:
        normalized = message.lower()
        return {
            "authenticate": any(token in normalized for token in ["authenticate", "auth", "verify gee", "gee login", "验证", "认证", "登录"]),
            "run": any(token in normalized for token in ["run", "start", "execute", "detect", "analyze", "分析", "执行", "运行", "识别"]),
            "status": any(token in normalized for token in ["status", "current", "summary", "show state", "当前", "状态", "参数", "配置"]),
        }

    def _is_reset(self, message: str) -> bool:
        normalized = message.lower()
        return any(token in normalized for token in ["reset", "clear", "restart", "重置", "清空", "重新开始"])

    def _extract_and_update_state(self, message: str, state: ConversationState) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}

        sensor = self._extract_sensor(message)
        if sensor and sensor != state.satellite:
            state.satellite = sensor
            updates["satellite"] = sensor

        method = self._extract_method(message)
        if method and method != state.method:
            state.method = method
            updates["method"] = method

        dates = self._extract_dates(message)
        if dates["date_start"] and dates["date_start"] != state.date_start:
            state.date_start = dates["date_start"]
            updates["date_start"] = dates["date_start"]
        if dates["date_end"] and dates["date_end"] != state.date_end:
            state.date_end = dates["date_end"]
            updates["date_end"] = dates["date_end"]

        region_geojson = self._extract_geojson(message)
        if region_geojson and region_geojson != state.region_geojson:
            state.region_geojson = region_geojson
            updates["region_geojson"] = region_geojson

        return updates

    def _extract_sensor(self, message: str) -> Optional[str]:
        sensor_aliases = {
            "viirs": "VIIRS",
            "modis": "MODIS",
            "landsat": "Landsat",
            "sentinel2": "Sentinel2",
            "sentinel-2": "Sentinel2",
            "sentinel 2": "Sentinel2",
        }
        normalized = message.lower()
        for alias, canonical in sensor_aliases.items():
            if alias in normalized:
                return canonical
        return None

    def _extract_method(self, message: str) -> Optional[str]:
        normalized = message.lower()
        if "quick fire" in normalized or "quick detection" in normalized or "快速" in normalized:
            return "quick_fire_detection"
        if "nbr" in normalized:
            return "NBR"
        if "evi" in normalized:
            return "EVI"
        return None

    def _extract_dates(self, message: str) -> Dict[str, Optional[str]]:
        matches = re.findall(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", message)
        normalized = [item.replace("/", "-") for item in matches]
        result = {"date_start": None, "date_end": None}
        if len(normalized) >= 1:
            result["date_start"] = normalized[0]
        if len(normalized) >= 2:
            result["date_end"] = normalized[1]
        return result

    def _extract_geojson(self, message: str) -> Optional[str]:
        json_block = self._find_json_object(message)
        if not json_block:
            return None

        try:
            parsed = json.loads(json_block)
        except Exception:
            return None

        if isinstance(parsed, dict) and parsed.get("type") in {"Polygon", "MultiPolygon", "Feature", "FeatureCollection"}:
            return json.dumps(parsed, ensure_ascii=False)
        return None

    def _find_json_object(self, text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        return None

    def _format_update_message(self, updated_fields: Dict[str, Any], state: ConversationState) -> str:
        readable = []
        if "satellite" in updated_fields:
            readable.append(f"satellite={state.satellite}")
        if "method" in updated_fields:
            readable.append(f"method={state.method}")
        if "date_start" in updated_fields:
            readable.append(f"date_start={state.date_start}")
        if "date_end" in updated_fields:
            readable.append(f"date_end={state.date_end}")
        if "region_geojson" in updated_fields:
            readable.append(self._text("region=updated", "区域=已更新"))

        prefix = self._text("Updated context:", "已更新上下文：")
        return f"{prefix} {', '.join(readable)}"

    def _authenticate(self, state: ConversationState) -> str:
        try:
            init_gee()
            state.gee_authenticated = True
            return self._text(
                "GEE authentication completed successfully.",
                "GEE 验证已完成。",
            )
        except Exception as exc:
            state.gee_authenticated = False
            return self._text(
                f"GEE authentication failed: {exc}",
                f"GEE 验证失败：{exc}",
            )

    def _format_status(self, state: ConversationState) -> str:
        region_ready = "yes" if state.region_geojson else "no"
        if self.language == "zh":
            region_ready = "是" if state.region_geojson else "否"

        return self._text(
            (
                "Current session status:\n"
                f"- GEE authenticated: {'yes' if state.gee_authenticated else 'no'}\n"
                f"- Satellite: {state.satellite or 'not set'}\n"
                f"- Method: {state.method or 'not set'}\n"
                f"- Start date: {state.date_start or 'not set'}\n"
                f"- End date: {state.date_end or 'not set'}\n"
                f"- Region loaded: {region_ready}"
            ),
            (
                "当前会话状态：\n"
                f"- GEE 已验证：{'是' if state.gee_authenticated else '否'}\n"
                f"- 卫星：{state.satellite or '未设置'}\n"
                f"- 方法：{state.method or '未设置'}\n"
                f"- 开始日期：{state.date_start or '未设置'}\n"
                f"- 结束日期：{state.date_end or '未设置'}\n"
                f"- 区域已加载：{region_ready}"
            ),
        )

    def _looks_like_full_task(self, message: str, state: ConversationState) -> bool:
        normalized = message.lower()
        has_execution_hint = any(token in normalized for token in ["for", "using", "use", "对", "使用"])
        return has_execution_hint and all(
            [state.satellite, state.method, state.date_start, state.date_end, state.region_geojson]
        )

    def _run_detection(self, state: ConversationState) -> Dict[str, Any]:
        missing = self._missing_fields(state)
        if missing:
            missing_text = ", ".join(missing)
            return {
                "ok": False,
                "result": None,
                "reply": self._text(
                    f"I am missing these required fields before running: {missing_text}.",
                    f"执行前还缺少这些必要参数：{missing_text}。",
                ),
            }

        if not state.gee_authenticated:
            auth_message = self._authenticate(state)
            if not state.gee_authenticated:
                return {"ok": False, "result": None, "reply": auth_message}

        try:
            if state.method == "quick_fire_detection":
                raw_result = run_quick_fire_detection_tool.invoke(
                    {
                        "region_geojson": state.region_geojson,
                        "date_start": state.date_start,
                        "date_end": state.date_end,
                        "sensor": state.satellite,
                    }
                )
            elif state.method == "NBR":
                raw_result = run_nbr_index_method_tool.invoke(
                    {
                        "region_geojson": state.region_geojson,
                        "date_start": state.date_start,
                        "date_end": state.date_end,
                        "sensor": state.satellite,
                    }
                )
            elif state.method == "EVI":
                raw_result = run_evi_method_tool.invoke(
                    {
                        "region_geojson": state.region_geojson,
                        "date_start": state.date_start,
                        "date_end": state.date_end,
                        "sensor": state.satellite,
                    }
                )
            else:
                raise ValueError(f"Unsupported method: {state.method}")

            result = json.loads(raw_result)
            return {
                "ok": result.get("status") == "ok",
                "result": result,
                "reply": self._format_run_reply(result),
            }
        except Exception as exc:
            return {
                "ok": False,
                "result": {"status": "error", "message": str(exc)},
                "reply": self._text(
                    f"Execution failed: {exc}",
                    f"执行失败：{exc}",
                ),
            }

    def _format_run_reply(self, result: Dict[str, Any]) -> str:
        if result.get("status") != "ok":
            return self._text(
                f"The detection task failed: {result.get('message', 'unknown error')}",
                f"识别任务失败：{result.get('message', '未知错误')}",
            )

        summary_lines = [
            self._text("Detection task completed.", "识别任务已完成。"),
            f"status: {result.get('status')}",
        ]

        for key in [
            "method_name",
            "sensor",
            "date_start",
            "date_end",
            "image_count",
            "candidate_pixel_count",
            "fire_pixel_count_estimate",
            "message",
        ]:
            if key in result:
                summary_lines.append(f"{key}: {result.get(key)}")

        return "\n".join(summary_lines)

    def _guidance_message(self, state: ConversationState) -> str:
        if not state.gee_authenticated:
            return self._text(
                "You can ask me to authenticate GEE first, or directly provide a full task with method, sensor, dates, and GeoJSON region.",
                "你可以先让我执行 GEE 验证，或者直接用一条完整指令告诉我方法、卫星、时间和 GeoJSON 区域。",
            )

        return self._text(
            "I can continue from the current context. You can ask me to show status, reset, or run a detection task with dates and GeoJSON.",
            "我可以基于当前上下文继续。你可以让我查看状态、重置，或者输入带日期和 GeoJSON 的识别任务。",
        )

    def _missing_fields(self, state: ConversationState) -> list[str]:
        field_labels = {
            "satellite": self._text("satellite", "卫星"),
            "method": self._text("method", "方法"),
            "date_start": self._text("start date", "开始日期"),
            "date_end": self._text("end date", "结束日期"),
            "region_geojson": self._text("GeoJSON region", "GeoJSON 区域"),
        }

        missing = []
        for field_name in ["satellite", "method", "date_start", "date_end", "region_geojson"]:
            if not getattr(state, field_name):
                missing.append(field_labels[field_name])
        return missing
