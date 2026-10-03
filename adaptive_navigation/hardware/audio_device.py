"""
Wearable Audio Device Abstraction.

Provides a unified interface for audio output targets:
- SimulationAudioDevice: Mock/Simulation logging without hardware dependency.
- SystemAudioDevice: Standard system speakers / line-out via TTSEngine.
- WearableBluetoothAudioDevice: Wearable earbud / headset audio sink.
"""

from abc import ABC, abstractmethod
import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AudioDevice(ABC):
    """Abstract base class for audio output destinations."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection with the audio target."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the audio target."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if device is actively connected."""
        pass

    @abstractmethod
    def play_audio(self, text: str, priority: str = "MEDIUM", timestamp: float = 0.0) -> bool:
        """Plays or logs synthesized audio message."""
        pass

    @abstractmethod
    def stop_audio(self) -> None:
        """Immediately halts ongoing audio output."""
        pass

    @abstractmethod
    def get_status(self) -> dict:
        """Returns device diagnostic status."""
        pass


class SimulationAudioDevice(AudioDevice):
    """
    Simulation audio device that records and logs messages without requiring physical hardware.
    """

    def __init__(self, device_name: str = "Virtual Wearable Sink") -> None:
        self.device_name = device_name
        self._connected: bool = True
        self.played_messages: List[Dict[str, any]] = []

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def play_audio(self, text: str, priority: str = "MEDIUM", timestamp: float = 0.0) -> bool:
        if not self._connected or not text:
            return False

        t_now = time.time()
        latency_ms = (t_now - timestamp) * 1000.0 if timestamp > 0 else 0.0

        entry = {
            "text": text,
            "priority": priority,
            "timestamp": t_now,
            "latency_ms": latency_ms,
        }
        self.played_messages.append(entry)
        print(f"[AUDIO SIMULATION] \"{text}\" (Priority: {priority}, Latency: {latency_ms:.1f}ms)")
        return True

    def stop_audio(self) -> None:
        pass

    def get_status(self) -> dict:
        return {
            "device_name": self.device_name,
            "device_type": "simulation",
            "connected": self._connected,
            "messages_played": len(self.played_messages),
        }


class SystemAudioDevice(AudioDevice):
    """
    Audio device routing speech to local system speakers via TTSEngine.
    """

    def __init__(self, tts_engine=None, device_name: str = "System Audio Speaker") -> None:
        self.tts = tts_engine
        self.device_name = device_name
        self._connected: bool = bool(tts_engine and tts_engine.is_available())

    def connect(self) -> bool:
        if self.tts and self.tts.is_available():
            self._connected = True
            return True
        self._connected = False
        return False

    def disconnect(self) -> None:
        self._connected = False
        if self.tts:
            self.tts.stop()

    def is_connected(self) -> bool:
        return self._connected and (self.tts is not None and self.tts.is_available())

    def play_audio(self, text: str, priority: str = "MEDIUM", timestamp: float = 0.0) -> bool:
        if not self.is_connected() or not text:
            return False
        return self.tts.speak(text, priority=priority)

    def stop_audio(self) -> None:
        if self.tts:
            self.tts.stop()

    def get_status(self) -> dict:
        return {
            "device_name": self.device_name,
            "device_type": "system_speaker",
            "connected": self.is_connected(),
            "backend": getattr(self.tts, "active_backend", "none") if self.tts else "none",
        }


class WearableBluetoothAudioDevice(AudioDevice):
    """
    Wearable Bluetooth earbud abstraction supporting connection management and audio playback.
    """

    def __init__(self, tts_engine=None, device_name: str = "Wearable Bluetooth Earbuds") -> None:
        self.tts = tts_engine
        self.device_name = device_name
        self._connected: bool = True
        self.messages_count: int = 0

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False
        if self.tts:
            self.tts.stop()

    def is_connected(self) -> bool:
        return self._connected

    def play_audio(self, text: str, priority: str = "MEDIUM", timestamp: float = 0.0) -> bool:
        if not self._connected or not text:
            return False

        self.messages_count += 1
        t_now = time.time()
        latency_ms = (t_now - timestamp) * 1000.0 if timestamp > 0 else 0.0
        print(f"[WEARABLE EARBUD] \"{text}\" (Priority: {priority}, Latency: {latency_ms:.1f}ms)")

        if self.tts and self.tts.is_available():
            return self.tts.speak(text, priority=priority)
        return True

    def stop_audio(self) -> None:
        if self.tts:
            self.tts.stop()

    def get_status(self) -> dict:
        return {
            "device_name": self.device_name,
            "device_type": "wearable_bluetooth",
            "connected": self._connected,
            "messages_sent": self.messages_count,
        }
