from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(include_in_schema=False)

_PAGE = Path(__file__).resolve().parents[2] / "static" / "index.html"


@router.get("/")
def index() -> FileResponse:
    # No caching, so a browser never keeps showing an older version of the page.
    return FileResponse(_PAGE, headers={"Cache-Control": "no-store"})
