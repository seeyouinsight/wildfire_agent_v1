import ee
from agent.config import GEE_PROJECT_ID

_GEE_INITIALIZED = False


def init_gee() -> None:
    global _GEE_INITIALIZED
    if _GEE_INITIALIZED:
        return

    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        _GEE_INITIALIZED = True
        print(f"GEE initialized with project: {GEE_PROJECT_ID}")
        return
    except Exception:
        pass

    ee.Authenticate(auth_mode="notebook")
    ee.Initialize(project=GEE_PROJECT_ID)
    _GEE_INITIALIZED = True
    print(f"GEE authenticated and initialized with project: {GEE_PROJECT_ID}")