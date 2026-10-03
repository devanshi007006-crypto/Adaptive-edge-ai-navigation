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
