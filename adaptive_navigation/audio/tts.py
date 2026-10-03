"""
Modular Offline-First Text-to-Speech (TTS) Engine.

Supports multiple pluggable backends:
1. 'pyttsx3': Offline native SAPI5 speech synthesis via pyttsx3.
2. 'powershell': Native Windows System.Speech SAPI invocation via background process.
3. 'dummy' / 'mock': Silent non-blocking fallback for headless testing.
4. 'auto': Automatically discovers the best available offline backend.

Features:
- Completely non-blocking asynchronous speech worker thread.
- Small bounded priority queue with queue overflow prevention.
- Priority preemption (e.g. CRITICAL interrupts ongoing CAUTION).
- Failsafe execution: never crashes the perception or navigation pipeline.
"""

from collections import deque
import logging
import queue
import subprocess
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Modular, offline-first, non-blocking text-to-speech engine.
    """

    PRIORITY_LEVELS = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        audio_cfg = self.config.get("audio", {})

        self.enabled = bool(audio_cfg.get("enabled", True))
        self.backend_choice = audio_cfg.get("backend", "auto").lower()
        self.language = audio_cfg.get("language", "en")
        self.volume = float(audio_cfg.get("volume", 1.0))
        self.allow_interrupt = bool(audio_cfg.get("allow_priority_interrupt", True))
        self.max_queue_size = int(audio_cfg.get("max_queue_size", 2))

        self.active_backend: str = "none"
        self.audio_available: bool = False
        self._pyttsx3_engine = None

        # Thread-safe queue and worker
        self._speech_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=self.max_queue_size + 1)
        self._current_priority: int = 0
        self._is_speaking: bool = False
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        if self.enabled:
            self._initialize_backend()
            self._start_worker()

    def _initialize_backend(self) -> None:
        """Select and initialize the chosen speech backend."""
        # 1. Try pyttsx3
        if self.backend_choice in ("auto", "pyttsx3"):
            try:
                import pyttsx3
                self._pyttsx3_engine = pyttsx3.init()
                self._pyttsx3_engine.setProperty("volume", self.volume)
                self.active_backend = "pyttsx3"
                self.audio_available = True
                logger.info("[TTSEngine] Initialized offline pyttsx3 backend.")
                return
            except Exception as e:
                logger.warning(f"[TTSEngine] pyttsx3 init failed: {e}")

        # 2. Try PowerShell System.Speech (Windows native)
        if self.backend_choice in ("auto", "powershell"):
            try:
                res = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", "Add-Type -AssemblyName System.Speech; Write-Output 'OK'"],
                    capture_output=True, text=True, timeout=3.0
                )
                if "OK" in res.stdout:
                    self.active_backend = "powershell"
                    self.audio_available = True
                    logger.info("[TTSEngine] Initialized Windows System.Speech PowerShell backend.")
                    return
            except Exception as e:
                logger.warning(f"[TTSEngine] PowerShell System.Speech init failed: {e}")

        # 3. Fallback to dummy
        self.active_backend = "dummy"
        self.audio_available = True
        logger.info("[TTSEngine] Fallback to dummy non-blocking audio backend.")

    def _start_worker(self) -> None:
        """Starts background speech worker thread."""
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Worker loop processing prioritized speech utterances."""
        while not self._stop_event.is_set():
            try:
                priority_item = self._speech_queue.get(timeout=0.2)
                inv_priority, timestamp, text, prio_str = priority_item

                prio_val = -inv_priority
                self._current_priority = prio_val
                self._is_speaking = True

                self._execute_speech(text)

                self._is_speaking = False
                self._current_priority = 0
                self._speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[TTSEngine] Error in speech worker: {e}")
                self._is_speaking = False
                self._current_priority = 0

    def _execute_speech(self, text: str) -> None:
        """Executes speech using active backend."""
        if not text or not self.audio_available:
            return

        try:
            if self.active_backend == "pyttsx3" and self._pyttsx3_engine is not None:
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()

            elif self.active_backend == "powershell":
                ps_cmd = f'Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak("{text}")'
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    capture_output=True, timeout=5.0
                )

            elif self.active_backend == "dummy":
                time.sleep(0.05)

        except Exception as e:
            logger.error(f"[TTSEngine] Speech execution failed: {e}")

    def speak(self, text: str, priority: str = "MEDIUM") -> bool:
        """
        Dispatches an utterance to be spoken asynchronously.
        Higher priority warnings may interrupt lower-priority ongoing speech.
        """
        if not self.enabled or not self.audio_available or not text:
            return False

        prio_val = self.PRIORITY_LEVELS.get(priority.upper(), 2)

        # Check interruption
        if self._is_speaking and self.allow_interrupt:
            if prio_val > self._current_priority:
                self.stop()

        # Manage queue overflow
        if self._speech_queue.full():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except queue.Empty:
                pass

        try:
            self._speech_queue.put_nowait((-prio_val, time.time(), text, priority))
            return True
        except queue.Full:
            return False

    def stop(self) -> None:
        """Stops current speech and clears pending queue."""
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except queue.Empty:
                break

        if self.active_backend == "pyttsx3" and self._pyttsx3_engine is not None:
            try:
                self._pyttsx3_engine.stop()
            except Exception:
                pass

    def is_available(self) -> bool:
        """Returns True if a valid audio backend is active."""
        return self.enabled and self.audio_available

    def shutdown(self) -> None:
        """Shuts down background worker cleanly."""
        self._stop_event.set()
        self.stop()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
