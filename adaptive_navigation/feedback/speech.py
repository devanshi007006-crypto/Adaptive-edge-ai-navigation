from abc import ABC, abstractmethod
import threading
from typing import Optional

class OfflineTTSInterface(ABC):
    """Abstract interface for offline text-to-speech synthesis."""
    @abstractmethod
    def speak(self, text: str, interrupt: bool = False) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

class Pyttsx3SpeechEngine(OfflineTTSInterface):
    """Offline TTS engine using pyttsx3."""
    def __init__(self, rate: int = 175, volume: float = 1.0, voice_index: int = 0):
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self._engine = None

    def speak(self, text: str, interrupt: bool = False) -> None:
        # Concrete offline TTS execution in STEP 15
        pass

    def stop(self) -> None:
        pass
