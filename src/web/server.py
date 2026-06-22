"""
Simple static HTTP server to serve the ASV dashboard.

Serves the dashboard.html which connects directly to Firebase RTDB
for real-time telemetry and command sending. No FastAPI needed.
"""

import asyncio
import http.server
import os
import threading
from functools import partial

from src.config import WEB_DASHBOARD_PORT
from src.utils.logger import setup_logger

log = setup_logger("dashboard")


class DashboardServer:
    """Lightweight static file server for the dashboard."""

    def __init__(self, port: int = WEB_DASHBOARD_PORT):
        self._port = port
        self._static_dir = os.path.join(os.path.dirname(__file__), "static")
        self._server = None
        self._thread = None

    async def start(self):
        """Start the HTTP server in a background thread."""
        handler = partial(
            http.server.SimpleHTTPRequestHandler,
            directory=self._static_dir,
        )
        self._server = http.server.HTTPServer(("0.0.0.0", self._port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="dashboard-http",
        )
        self._thread.start()
        log.info("Dashboard server started at http://0.0.0.0:%d", self._port)

    def stop(self):
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
            log.info("Dashboard server stopped")

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}/dashboard.html"
