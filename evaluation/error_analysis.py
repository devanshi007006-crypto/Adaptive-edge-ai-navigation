"""
Systematic Error Analysis and Failure Categorization for Step 16 Research Evaluation.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json


@dataclass
class FailureCase:
    frame_id: int
    track_id: Optional[int]
    error_type: str
    ground_truth: Any
    prediction: Any
    possible_cause: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ErrorTypeSummary:
    error_type: str
    count: int
    percentage: float
    example: str
    possible_cause: str

    def to_dict(self) -> dict:
        return asdict(self)


class ErrorAnalyzer:
    """
    Categorizes, tallies, and investigates pipeline failures across all 13 canonical failure modes.
    """

    ERROR_DEFINITIONS = {
        "FALSE_DETECTION": ("Spurious bounding box without real object", "Background texture or reflection artifact"),
        "MISSED_DETECTION": ("Failed to detect actual physical obstacle", "Low contrast, partial occlusion, or extreme distance"),
        "ID_SWITCH": ("Track ID changed across consecutive observations", "Visual occlusion or sudden ego-motion jump"),
        "DEPTH_FAILURE": ("Depth error exceeds acceptable tolerance", "Monocular scale ambiguity or reflective surface"),
        "MOTION_FAILURE": ("Estimated motion contradicts true relative movement", "Sensor frame drop or noisy bounding box jitter"),
        "CAMERA_MOTION_FAILURE": ("Ego-motion compensation misattributes camera shift", "Insufficient static optical flow feature points"),
        "TTC_FAILURE": ("TTC calculation missing or invalid for closing target", "Low closing speed or unstable depth rate"),
        "RISK_MISCLASSIFICATION": ("Estimated risk level diverges from ground truth", "Over-reliance on noisy feature or path geometry error"),
        "RELIABILITY_MISMATCH": ("High reliability score assigned to erroneous prediction", "Uncalibrated evidence weights under domain shift"),
        "FALSE_WARNING": ("Audible warning triggered for safe scenario", "Transient risk spike before temporal stabilization"),
        "MISSED_WARNING": ("Imminent collision hazard failed to trigger alert", "Under-estimated risk or aggressive temporal suppression"),
        "INCORRECT_NAVIGATION": ("Avoidance recommendation directs user toward obstacle", "Corridor missegmentation or asymmetric clearance error"),
        "AUDIO_FAILURE": ("Voice warning dropped or delivery delayed", "Audio device disconnected or priority queue dropped"),
    }

    def __init__(self):
        self.failures: List[FailureCase] = []

    def record_failure(self, error_type: str, frame_id: int, track_id: Optional[int], ground_truth: Any, prediction: Any, possible_cause: Optional[str] = None, metadata: Optional[dict] = None) -> None:
        if error_type not in self.ERROR_DEFINITIONS:
            default_desc, default_cause = "Uncategorized anomaly", "Unknown etiology"
        else:
            default_desc, default_cause = self.ERROR_DEFINITIONS[error_type]

        cause = possible_cause if possible_cause else default_cause
        case = FailureCase(
            frame_id=frame_id,
            track_id=track_id,
            error_type=error_type,
            ground_truth=ground_truth,
            prediction=prediction,
            possible_cause=cause,
            metadata=metadata or {},
        )
        self.failures.append(case)

    def analyze(self) -> List[ErrorTypeSummary]:
        total_errors = len(self.failures)
        counts: Dict[str, int] = {k: 0 for k in self.ERROR_DEFINITIONS}
        examples: Dict[str, str] = {}

        for f in self.failures:
            counts[f.error_type] = counts.get(f.error_type, 0) + 1
            if f.error_type not in examples:
                examples[f.error_type] = f"Frame {f.frame_id}: GT={f.ground_truth} vs Pred={f.prediction}"

        summaries = []
        for err_type, (desc, def_cause) in self.ERROR_DEFINITIONS.items():
            cnt = counts[err_type]
            pct = (cnt / total_errors * 100.0) if total_errors > 0 else 0.0
            ex = examples.get(err_type, "No occurrences recorded.")
            summaries.append(ErrorTypeSummary(
                error_type=err_type,
                count=cnt,
                percentage=round(pct, 2),
                example=ex,
                possible_cause=def_cause,
            ))

        return summaries

    def export_failures_to_json(self, json_path: str) -> None:
        import os
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([f.to_dict() for f in self.failures], f, indent=2)
