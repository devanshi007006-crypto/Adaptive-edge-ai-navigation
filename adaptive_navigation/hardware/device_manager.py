"""
Hardware Device Manager & System Watchdog.

Supervises audio output hardware lifecycle, periodic connection health checks,
reconnection retry logic, and end-to-end subsystem health monitoring.
"""

from dataclasses import dataclass
import logging
import time
from typing import Dict, Optional, Tuple

from hardware.audio_device import (
    AudioDevice,
    SimulationAudioDevice,
    SystemAudioDevice,
    WearableBluetoothAudioDevice,
)

logger = logging.getLogger(__name__)


@dataclass
class DeviceStatus:
    """Diagnostic status for the selected audio output device."""
    connected: bool
    device_name: str
    device_type: str
    available: bool
    error: Optional[str]
    reconnect_attempts: int
    last_checked_time: float


@dataclass
class SystemStatus:
    """End-to-end health status across all pipeline stages."""
    camera_ok: bool
    detector_ok: bool
    tracker_ok: bool
    depth_ok: bool
    motion_ok: bool
    ttc_ok: bool
    risk_ok: bool
    reliability_ok: bool
    navigation_ok: bool
    tts_ok: bool
    audio_device_ok: bool
    overall_healthy: bool


class DeviceManager:
    """
    Manages audio output device lifecycle, simulation fallback,
    and end-to-end system watchdog monitoring.
    """

    def __init__(self, config: Optional[dict] = None, tts_engine=None) -> None:
        self.config = config or {}
        hw_cfg = self.config.get("hardware", {})

        self.enabled = bool(hw_cfg.get("enabled", True))
        self.simulation_mode = bool(hw_cfg.get("simulation_mode", True))
        self.device_type = hw_cfg.get("device_type", "auto").lower()
        self.reconnect_interval = float(hw_cfg.get("reconnect_interval_seconds", 5.0))

        self.tts = tts_engine
        self.device: Optional[AudioDevice] = None
        self.reconnect_attempts: int = 0
        self.last_check_time: float = 0.0

        self._initialize_device()

    def _initialize_device(self) -> None:
        """Instantiate device based on configuration."""
        if self.simulation_mode or self.device_type == "simulation":
            self.device = SimulationAudioDevice()
            logger.info("[DeviceManager] Initialized SimulationAudioDevice.")
        elif self.device_type == "wearable":
            self.device = WearableBluetoothAudioDevice(tts_engine=self.tts)
            logger.info("[DeviceManager] Initialized WearableBluetoothAudioDevice.")
        elif self.device_type in ("system", "auto"):
            if self.tts and self.tts.is_available():
                self.device = SystemAudioDevice(tts_engine=self.tts)
                logger.info("[DeviceManager] Initialized SystemAudioDevice.")
            else:
                self.device = SimulationAudioDevice("Fallback Simulation Sink")
                logger.info("[DeviceManager] Fallback to SimulationAudioDevice.")
        else:
            self.device = SimulationAudioDevice()

        if self.device:
            self.device.connect()
        self.last_check_time = time.time()

    def dispatch_audio(
        self,
        text: str,
        priority: str = "MEDIUM",
        decision_timestamp: float = 0.0
    ) -> Tuple[bool, float]:
        """
        Dispatches message to the active audio device and returns success status
        along with measured decision-to-audio latency (ms).
        """
        t_now = time.time()
        latency_ms = (t_now - decision_timestamp) * 1000.0 if decision_timestamp > 0 else 0.0

        if not self.enabled or not text:
            return False, latency_ms

        # Check connection with rate-limited retry
        if not self.is_connected():
            if (t_now - self.last_check_time) >= self.reconnect_interval:
                self.last_check_time = t_now
                self.reconnect_attempts += 1
                logger.info(f"[DeviceManager] Attempting device reconnect #{self.reconnect_attempts}...")
                if self.device:
                    self.device.connect()

        if not self.is_connected():
            logger.warning("[DeviceManager] AUDIO_DEVICE_UNAVAILABLE: message suppressed.")
            return False, latency_ms

        success = self.device.play_audio(text, priority=priority, timestamp=decision_timestamp)
        return success, latency_ms

    def is_connected(self) -> bool:
        """Check if active device is connected."""
        return bool(self.device and self.device.is_connected())

    def get_status(self) -> DeviceStatus:
        """Returns diagnostic status of the audio hardware device."""
        t_now = time.time()
        conn = self.is_connected()
        dev_info = self.device.get_status() if self.device else {}

        return DeviceStatus(
            connected=conn,
            device_name=dev_info.get("device_name", "None"),
            device_type=dev_info.get("device_type", "None"),
            available=conn,
            error=None if conn else "AUDIO_DEVICE_UNAVAILABLE",
            reconnect_attempts=self.reconnect_attempts,
            last_checked_time=self.last_check_time,
        )

    def check_system_health(
        self,
        camera_ok: bool,
        detector_ok: bool,
        tracker_ok: bool,
        depth_ok: bool,
        motion_ok: bool,
        ttc_ok: bool,
        risk_ok: bool,
        reliability_ok: bool,
        navigation_ok: bool,
        tts_ok: bool,
    ) -> SystemStatus:
        """
        Consolidates operational status across all pipeline stages.
        """
        audio_ok = self.is_connected()
        overall = bool(
            camera_ok
            and detector_ok
            and tracker_ok
            and depth_ok
            and reliability_ok
            and navigation_ok
        )

        return SystemStatus(
            camera_ok=camera_ok,
            detector_ok=detector_ok,
            tracker_ok=tracker_ok,
            depth_ok=depth_ok,
            motion_ok=motion_ok,
            ttc_ok=ttc_ok,
            risk_ok=risk_ok,
            reliability_ok=reliability_ok,
            navigation_ok=navigation_ok,
            tts_ok=tts_ok,
            audio_device_ok=audio_ok,
            overall_healthy=overall,
        )
