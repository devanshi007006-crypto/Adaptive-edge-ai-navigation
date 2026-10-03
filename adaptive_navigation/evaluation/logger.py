from dataclasses import dataclass, asdict
from typing import Optional, List
import csv
import os

@dataclass
class FrameLogRecord:
    """Minimum required log schema for evaluation."""
    timestamp: float
    frame_number: int
    fps: float
    latency_ms: float
    object_count: int
    highest_risk_score: float
    highest_risk_level: str
    navigation_decision: str
    warning_generated: Optional[str]
    warning_suppressed: Optional[str]
    warning_priority: Optional[str]

class NavigationLogger:
    """Logs per-frame evaluation records to structured CSV format."""
    def __init__(self, output_dir: str = "data/results", filename: str = "navigation_log.csv"):
        self.output_dir = output_dir
        self.filepath = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        self.records: List[FrameLogRecord] = []

    def log_frame(self, record: FrameLogRecord) -> None:
        self.records.append(record)


import json

@dataclass
class ExperimentEventRecord:
    """Detailed research experiment event schema for Step 15."""
    timestamp: float
    frame_id: int
    track_id: Optional[int]
    object_class: str
    depth: Optional[float]
    depth_type: str
    raw_motion: Optional[tuple]
    compensated_motion: Optional[tuple]
    ttc: Optional[float]
    ttc_state: str
    risk_score: float
    risk_level: str
    reliability_score: float
    reliability_level: str
    warning_state: str
    navigation_state: str
    message: str
    audio_status: str
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentLogger:
    """Appends structured experiment event logs to JSONL format."""
    def __init__(self, log_file: str = "data/experiment_log.jsonl") -> None:
        self.log_file = log_file
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log_event(self, event: ExperimentEventRecord) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
