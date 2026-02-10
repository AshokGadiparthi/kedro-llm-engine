"""
API Router — Combines all endpoint groups under /agent.

Original (7 endpoints):     /api/v1/agent/{insights,ask,validate,recommend,feedback,readiness,health}
Enhanced (16 endpoints):    /api/v1/agent/enhanced/*, /api/v1/agent/feedback/*, /api/v1/agent/cost-matrices/*, etc.
"""

from fastapi import APIRouter

from app.api.v1.agent import router as agent_router
from app.api.v1.enhanced import router as enhanced_router
from app.api.v1.eda_endpoints import router as eda_router

api_router = APIRouter()

api_router.include_router(
    agent_router,
    prefix="/agent",
    tags=["ML Expert Agent — Core (v3)"],
)

api_router.include_router(
    enhanced_router,
    prefix="/agent",
    tags=["ML Expert Agent — Enhanced (v4.0)"],
)

api_router.include_router(
    eda_router,
    prefix="/agent",
    tags=["ML Expert Agent — EDA Enhancements"],
)
