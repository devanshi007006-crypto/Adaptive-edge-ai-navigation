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
        self.rate = int(audio_cfg.get("rate", 180))
        self.allow_interrupt = bool(audio_cfg.get("allow_priority_interrupt", True))
        self.max_queue_size = int(audio_cfg.get("max_queue_size", 2))

        self.active_backend: str = "none"
        self.audio_available: bool = False
        self.last_error: str = ""
        self.selected_voice: str = "Default"
        self._pyttsx3_engine = None

        # Thread-safe queue and worker
        self._speech_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=self.max_queue_size + 1)
        self._current_priority: int = 0
        self._is_speaking: bool = False
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        if self.enabled:
            self._start_worker()

    def _start_worker(self) -> None:
        """Starts background speech worker thread."""
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Worker loop processing prioritized speech utterances inside worker thread."""
        # 1. Initialize COM on worker thread (Windows SAPI5 requirement)
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        # 2. Initialize Backend inside worker thread context
        if self.backend_choice in ("auto", "pyttsx3"):
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("volume", self.volume)
                engine.setProperty("rate", self.rate)
                voices = engine.getProperty("voices")
                if voices:
                    # Select first available English voice safely
                    eng_voice = next((v for v in voices if "EN" in v.id.upper() or "ENGLISH" in v.name.upper()), voices[0])
                    engine.setProperty("voice", eng_voice.id)
                    self.selected_voice = eng_voice.name
                self._pyttsx3_engine = engine
                self.active_backend = "pyttsx3"
                self.audio_available = True
                logger.info(f"[TTSEngine] Worker thread initialized pyttsx3 backend (Voice: {self.selected_voice}).")
            except Exception as e:
                logger.warning(f"[TTSEngine] pyttsx3 init in worker thread failed: {e}")
                self.last_error = str(e)

        if not self.audio_available and self.backend_choice in ("auto", "powershell"):
            try:
                res = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", "Add-Type -AssemblyName System.Speech; Write-Output 'OK'"],
                    capture_output=True, text=True, timeout=3.0
                )
                if "OK" in res.stdout:
                    self.active_backend = "powershell"
                    self.audio_available = True
                    self.selected_voice = "Windows System.Speech (PowerShell)"
                    logger.info("[TTSEngine] Initialized Windows System.Speech PowerShell backend.")
            except Exception as e:
                logger.warning(f"[TTSEngine] PowerShell System.Speech init failed: {e}")
                self.last_error = str(e)

        if not self.audio_available:
            self.active_backend = "dummy"
            self.audio_available = True
            self.selected_voice = "Dummy Silent Fallback"
            logger.info("[TTSEngine] Fallback to dummy non-blocking audio backend.")

        # Main speech processing loop
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
                self.last_error = str(e)
                self._is_speaking = False
                self._current_priority = 0

        # Cleanup COM on worker thread termination
        if self._pyttsx3_engine is not None:
            try:
                self._pyttsx3_engine.stop()
            except Exception:
                pass

        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:
            pass

    def _execute_speech(self, text: str) -> None:
        """Executes speech using active backend inside worker thread."""
        if not text or not self.audio_available:
            return

        try:
            if self.active_backend == "pyttsx3" and self._pyttsx3_engine is not None:
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()

            elif self.active_backend == "powershell":
                vol_int = int(self.volume * 100)
                ps_cmd = f'Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Volume = {vol_int}; $s.Speak("{text}")'
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    capture_output=True, timeout=5.0
                )

            elif self.active_backend == "dummy":
                time.sleep(0.05)

        except Exception as e:
            logger.error(f"[TTSEngine] Speech execution failed: {e}")
            self.last_error = str(e)

    def speak(self, text: str, priority: str = "MEDIUM") -> bool:
        """
        Dispatches an utterance to be spoken asynchronously.
        Higher priority warnings interrupt lower-priority ongoing speech and flush obsolete items.
        """
        if not self.enabled or not self.audio_available or not text:
            return False

        prio_val = self.PRIORITY_LEVELS.get(priority.upper(), 2)

        # Single Active Queue Policy: Higher priority flushes lower priority queued items
        if prio_val > self._current_priority:
            while not self._speech_queue.empty():
                try:
                    self._speech_queue.get_nowait()
                    self._speech_queue.task_done()
                except queue.Empty:
                    break
            if self.allow_interrupt and self.active_backend == "pyttsx3" and self._pyttsx3_engine:
                try:
                    self._pyttsx3_engine.stop()
                except Exception:
                    pass

        # Manage queue overflow
        if self._speech_queue.full():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except queue.Empty:
                pass

        # Prevent duplicate identical utterances from queueing up
        with self._speech_queue.mutex:
            for item in self._speech_queue.queue:
                if len(item) > 2 and item[2] == text:
                    return False

        try:
            self._speech_queue.put_nowait((-prio_val, time.time(), text, priority))
            return True
        except queue.Full:
            return False

    def is_speaking(self) -> bool:
        """Returns True if an utterance is actively being spoken or queued."""
        return self._is_speaking or not self._speech_queue.empty()

    def get_status_text(self) -> str:
        """Returns current high-level status string for GUI indicators."""
        if self.last_error:
            return f"ERROR ({self.last_error[:20]})"
        if not self.audio_available:
            return "DISABLED"
        if self._is_speaking:
            return "SPEAKING"
        if not self._speech_queue.empty():
            return "QUEUED"
        return "READY"

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
