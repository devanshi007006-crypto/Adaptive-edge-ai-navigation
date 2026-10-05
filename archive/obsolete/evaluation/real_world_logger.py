"""
Real-World Data Logger for Adaptive Edge-AI Navigation System.
Supports structured logging of real-world trials, pilot testing, and user-centric validation.
Complies with Step 18 specification.
"""

import os
import csv
import json
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

@dataclass
class RealWorldEventRecord:
    timestamp: float                       # UNIX epoch timestamp
    test_id: str                          # Identifier (e.g. RW_001)
    frame_id: int                         # Sequential frame counter
    track_id: Optional[int]               # Object tracking ID
    object_class: str                     # Detected class (e.g. 'person', 'chair')
    depth: float                          # Metric depth estimate (meters)
    motion: str                           # 'approaching', 'receding', 'stationary', 'lateral', 'unknown'
    TTC: float                            # Time-to-collision (seconds, float('inf') if none)
    risk_score: float                     # Risk value [0.0, 1.0]
    risk_level: str                       # 'NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    reliability_score: float              # Perception reliability score [0.0, 1.0]
    reliability_level: str                # 'LOW', 'MEDIUM', 'HIGH'
    path_relevance: str                   # 'IN_PATH', 'MARGINAL', 'OUT_OF_PATH', 'UNKNOWN'
    warning_state: str                    # 'NONE', 'INFORMATIONAL', 'CAUTION', 'CRITICAL'
    navigation_state: str                 # 'CLEAR', 'SLOW_DOWN', 'STEP_LEFT', 'STEP_RIGHT', 'STOP', 'UNKNOWN'
    message: str                          # Alert string generated
    audio_status: str                     # 'DISPATCHED', 'QUEUED', 'SUPPRESSED', 'MUTED', 'IDLE', 'FAILED'
    latency: float                        # End-to-end frame processing latency (ms)
    warning_latency: Optional[float] = None     # Hazard detection to audio dispatch latency (ms)
    nav_audio_latency: Optional[float] = None   # Path block to audio dispatch latency (ms)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d['TTC'] == float('inf'):
            d['TTC'] = 999.0
        return d

class RealWorldLogger:
    """
    In-memory and persistent logger for controlled real-world trials.
    Records per-frame obstacle events, decisions, and system latency metrics.
    """
    def __init__(self, log_dir: Optional[str] = None):
        self.records: List[RealWorldEventRecord] = []
        self.log_dir = log_dir or os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, record: RealWorldEventRecord):
        self.records.append(record)

    def log_event(self,
                  test_id: str,
                  frame_id: int,
                  track_id: Optional[int],
                  object_class: str,
                  depth: float,
                  motion: str,
                  TTC: float,
                  risk_score: float,
                  risk_level: str,
                  reliability_score: float,
                  reliability_level: str,
                  path_relevance: str,
                  warning_state: str,
                  navigation_state: str,
                  message: str,
                  audio_status: str,
                  latency: float,
                  warning_latency: Optional[float] = None,
                  nav_audio_latency: Optional[float] = None,
                  **kwargs) -> RealWorldEventRecord:
        rec = RealWorldEventRecord(
            timestamp=time.time(),
            test_id=test_id,
            frame_id=frame_id,
            track_id=track_id,
            object_class=object_class,
            depth=round(float(depth), 3),
            motion=motion,
            TTC=round(float(TTC), 2) if TTC != float('inf') else 999.0,
            risk_score=round(float(risk_score), 3),
            risk_level=risk_level,
            reliability_score=round(float(reliability_score), 3),
            reliability_level=reliability_level,
            path_relevance=path_relevance,
            warning_state=warning_state,
            navigation_state=navigation_state,
            message=message,
            audio_status=audio_status,
            latency=round(float(latency), 2),
            warning_latency=round(float(warning_latency), 2) if warning_latency is not None else None,
            nav_audio_latency=round(float(nav_audio_latency), 2) if nav_audio_latency is not None else None,
            extra_metadata=kwargs
        )
        self.records.append(rec)
        return rec

    def export_csv(self, filename: Optional[str] = None) -> str:
        filepath = filename or os.path.join(self.log_dir, "real_world_log.csv")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not self.records:
            return filepath
        
        fieldnames = [
            'timestamp', 'test_id', 'frame_id', 'track_id', 'object_class',
            'depth', 'motion', 'TTC', 'risk_score', 'risk_level',
            'reliability_score', 'reliability_level', 'path_relevance',
            'warning_state', 'navigation_state', 'message', 'audio_status',
            'latency', 'warning_latency', 'nav_audio_latency'
        ]
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in self.records:
                writer.writerow(r.to_dict())
        return filepath

    def export_json(self, filename: Optional[str] = None) -> str:
        filepath = filename or os.path.join(self.log_dir, "real_world_log.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = [r.to_dict() for r in self.records]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return filepath

    def clear(self):
        self.records.clear()

    def get_summary(self) -> Dict[str, Any]:
        if not self.records:
            return {"total_records": 0}
        latencies = [r.latency for r in self.records]
        warning_lats = [r.warning_latency for r in self.records if r.warning_latency is not None]
        nav_lats = [r.nav_audio_latency for r in self.records if r.nav_audio_latency is not None]
        return {
            "total_records": len(self.records),
            "test_ids": sorted(list(set(r.test_id for r in self.records))),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "avg_warning_latency_ms": round(sum(warning_lats) / len(warning_lats), 2) if warning_lats else None,
            "avg_nav_audio_latency_ms": round(sum(nav_lats) / len(nav_lats), 2) if nav_lats else None,
            "total_audio_dispatched": sum(1 for r in self.records if r.audio_status == "DISPATCHED"),
            "total_audio_suppressed": sum(1 for r in self.records if r.audio_status == "SUPPRESSED")
        }
