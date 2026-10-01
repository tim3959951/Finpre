"""ASGI entry point: `uvicorn fintech_agent.api.main:app`."""
from .app import create_app

app = create_app()
