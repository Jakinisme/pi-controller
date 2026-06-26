"""
RTSP Camera Streamer for ASV.

Manages a camera stream (USB cam or Pi Camera) and exposes it as:
  - RTSP stream (for VLC or native RTSP clients)
  - HLS stream (for React dashboard via HLS.js in browser)

Architecture:
  Camera (USB/Pi) --> ffmpeg --> HLS segments (served via HTTP)
                              |
                              --> RTSP stream (via mediamtx/go2rtc, optional)

The stream URL is pushed to Firebase so the React dashboard knows where to connect.

Requirements on RPi:
  - ffmpeg with libx264 or h264_v4l2m2m encoder
  - A simple HTTP server (we spawn one) to serve HLS segments
"""

import asyncio
import os

import time
from typing import Optional

from src.config import (
    CAMERA_DEVICE,
    CAMERA_RESOLUTION,
    CAMERA_FPS,
    CAMERA_BITRATE,
    CAMERA_HLS_SEGMENT_TIME,
    CAMERA_HLS_OUTPUT_DIR,
    CAMERA_HTTP_PORT,
    CAMERA_ENABLED,
)
from src.utils.logger import setup_logger

log = setup_logger("camera")


class RTSPStreamer:
    """Manages ffmpeg-based camera streaming to HLS."""

    def __init__(
        self,
        device: str = CAMERA_DEVICE,
        resolution: str = CAMERA_RESOLUTION,
        fps: int = CAMERA_FPS,
        bitrate: str = CAMERA_BITRATE,
        hls_segment_time: int = CAMERA_HLS_SEGMENT_TIME,
        hls_output_dir: str = CAMERA_HLS_OUTPUT_DIR,
        http_port: int = CAMERA_HTTP_PORT,
        enabled: bool = CAMERA_ENABLED,
    ):
        self._device = device
        self._resolution = resolution
        self._fps = fps
        self._bitrate = bitrate
        self._hls_segment_time = hls_segment_time
        self._hls_dir = hls_output_dir
        self._http_port = http_port
        self._enabled = enabled

        self._ffmpeg_proc: Optional[asyncio.subprocess.Process] = None
        self._http_proc: Optional[asyncio.subprocess.Process] = None
        self._running = False

        # The HLS playlist URL that React dashboard will use
        self.stream_url = ""

    @staticmethod
    def _compute_bufsize(bitrate: str) -> str:
        """Parse a bitrate string (e.g. '800k', '1M') and return 2x bufsize."""
        br = bitrate.strip().lower()
        if br.endswith("k"):
            value = int(br[:-1]) * 2
            return f"{value}k"
        elif br.endswith("m"):
            value = float(br[:-1]) * 2
            return f"{value}M"
        else:
            # Raw numeric (bits/s)
            value = int(br) * 2
            return str(value)

    def _build_ffmpeg_cmd(self) -> list:
        """Build the ffmpeg command for camera -> HLS."""
        width, height = self._resolution.split("x")

        cmd = [
            "ffmpeg",
            # Input: V4L2 camera (USB cam or Pi Cam via libcamera)
            "-f", "v4l2",
            "-framerate", str(self._fps),
            "-video_size", self._resolution,
            "-i", self._device,
            # Video encoding: hardware-accelerated on RPi if available, fallback to libx264
            "-c:v", "libx264",       # Use h264_v4l2m2m for HW accel if supported
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-b:v", self._bitrate,
            "-maxrate", self._bitrate,
            "-bufsize", self._compute_bufsize(self._bitrate),
            "-g", str(self._fps * 2),  # Keyframe every 2 seconds
            # HLS output
            "-f", "hls",
            "-hls_time", str(self._hls_segment_time),
            "-hls_list_size", "5",          # Keep 5 segments in playlist
            "-hls_flags", "delete_segments+append_list",
            "-hls_segment_filename", os.path.join(self._hls_dir, "segment_%05d.ts"),
            os.path.join(self._hls_dir, "stream.m3u8"),
        ]
        return cmd

    def _build_http_server_cmd(self) -> list:
        """Build a simple Python HTTP server command to serve HLS files."""
        return [
            "python3", "-m", "http.server",
            str(self._http_port),
            "--directory", self._hls_dir,
        ]

    async def start(self, rpi_ip: str = "0.0.0.0") -> bool:
        """Start the camera stream and HTTP server.
        
        Args:
            rpi_ip: The RPi's IP address on the MiFi network.
                    Used to construct the stream URL for the dashboard.
        
        Returns:
            True if started successfully.
        """
        if not self._enabled:
            log.info("Camera streamer disabled")
            return False

        # Create HLS output directory
        os.makedirs(self._hls_dir, exist_ok=True)

        # Clean old segments
        for f in os.listdir(self._hls_dir):
            if f.endswith(".ts") or f.endswith(".m3u8"):
                os.remove(os.path.join(self._hls_dir, f))

        # Start ffmpeg
        ffmpeg_cmd = self._build_ffmpeg_cmd()
        log.info("Starting ffmpeg: %s", " ".join(ffmpeg_cmd))

        try:
            self._ffmpeg_proc = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            log.error("ffmpeg not found. Install: sudo apt install ffmpeg")
            return False
        except Exception as e:
            log.error("ffmpeg start failed: %s", e)
            return False

        # Start HTTP server for HLS
        http_cmd = self._build_http_server_cmd()
        try:
            self._http_proc = await asyncio.create_subprocess_exec(
                *http_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            log.error("HTTP server start failed: %s", e)

        self._running = True
        actual_ip = rpi_ip if rpi_ip != "0.0.0.0" else "localhost"
        self.stream_url = f"http://{actual_ip}:{self._http_port}/stream.m3u8"
        log.info("Camera stream started: %s", self.stream_url)

        # Monitor ffmpeg stderr in background
        asyncio.create_task(self._monitor_ffmpeg())
        return True

    async def _monitor_ffmpeg(self):
        """Monitor ffmpeg output for errors."""
        if not self._ffmpeg_proc or not self._ffmpeg_proc.stderr:
            return
        try:
            while self._running:
                line = await self._ffmpeg_proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="ignore").strip()
                if "error" in text.lower() or "fatal" in text.lower():
                    log.error("ffmpeg: %s", text)
        except Exception:
            pass

    async def stop(self):
        """Stop the stream and HTTP server."""
        self._running = False

        if self._ffmpeg_proc and self._ffmpeg_proc.returncode is None:
            self._ffmpeg_proc.terminate()
            try:
                await asyncio.wait_for(self._ffmpeg_proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._ffmpeg_proc.kill()
            log.info("ffmpeg stopped")

        if self._http_proc and self._http_proc.returncode is None:
            self._http_proc.terminate()
            try:
                await asyncio.wait_for(self._http_proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self._http_proc.kill()
            log.info("HTTP server stopped")

        # Clean up HLS segments
        if os.path.isdir(self._hls_dir):
            for f in os.listdir(self._hls_dir):
                if f.endswith(".ts") or f.endswith(".m3u8"):
                    os.remove(os.path.join(self._hls_dir, f))

        log.info("Camera streamer stopped")

    @property
    def is_running(self) -> bool:
        return self._running and self._ffmpeg_proc is not None and self._ffmpeg_proc.returncode is None

    async def health_check(self) -> bool:
        """Check if the stream is healthy."""
        if not self._ffmpeg_proc or self._ffmpeg_proc.returncode is not None:
            return False
        # Check if m3u8 playlist exists and is recent
        m3u8_path = os.path.join(self._hls_dir, "stream.m3u8")
        if not os.path.exists(m3u8_path):
            return False
        age = time.time() - os.path.getmtime(m3u8_path)
        return age < 10  # Playlist updated within last 10 seconds
