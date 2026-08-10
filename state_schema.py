from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    user_query: str
    region_text: str
    date_start: str
    date_end: str
    sensor_preference: str
    output_need: str

    parsed_region_type: str
    parsed_region_value: Dict[str, Any]
    task_type: str
    analysis_mode: str
    parsed_ok: bool

    workflow_id: str
    workflow_reason: str
    sensor_selected: str

    method_selected: str
    method_params: Dict[str, Any]

    gee_collection_id: str
    gee_task_id: str
    gee_status: str
    gee_message: str

    result_summary: Dict[str, Any]
    result_links: List[str]
    result_stats: Dict[str, Any]
    evaluation_result: Dict[str, Any]

    error_message: str
    logs: List[str]


def create_initial_state(
    user_query: str,
    region_text: str,
    date_start: str,
    date_end: str,
    sensor_preference: str = "",
    output_need: str = "summary"
) -> AgentState:
    return {
        "user_query": user_query,
        "region_text": region_text,
        "date_start": date_start,
        "date_end": date_end,
        "sensor_preference": sensor_preference,
        "output_need": output_need,
        "method_params": {},
        "result_summary": {},
        "result_links": [],
        "result_stats": {},
        "evaluation_result": {},
        "logs": []
    }