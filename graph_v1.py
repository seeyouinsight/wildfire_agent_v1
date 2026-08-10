from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from agent.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from agent.tools_gee import (
    run_evi_method_tool,
    run_nbr_index_method_tool,
    run_quick_fire_detection_tool,
)


def build_agent():
    model = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    return create_agent(
        model=model,
        tools=[
            run_quick_fire_detection_tool,
            run_nbr_index_method_tool,
            run_evi_method_tool,
        ],
        system_prompt=(
            "You are a geospatial fire detection assistant. "
            "Use Earth Engine tools when the user asks to analyze fire hotspots "
            "for a region and time range. Supported methods are quick fire product "
            "detection, NBR index analysis, and EVI index analysis. "
            "Ask for missing GeoJSON region, start date, or end date before running. "
            "Return concise, clear summaries in the user's language."
        ),
    )
