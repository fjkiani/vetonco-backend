"""VetOnco — Health check router."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "module": "vetonco", "version": "1.0.0"}
