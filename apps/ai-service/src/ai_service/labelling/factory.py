import httpx

from ai_service.core.config import Settings
from ai_service.labelling.system_one import SystemOneLabeller


def build_labeller(settings: Settings, client: httpx.AsyncClient) -> SystemOneLabeller:
    """Pick the decision model from config; it serves both the DecisionPort and ChoicePort."""
    if settings.labeller == "laya":
        return SystemOneLabeller(
            "laya", client, settings.laya_url, settings.laya_model, settings.laya_api_key
        )
    return SystemOneLabeller("strands", client, settings.decider_url)
