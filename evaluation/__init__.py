"""
Evaluation, Benchmarking & Experimental Validation Suite (Step 16-18).
"""

from .ground_truth import (
    GroundTruthBBox,
    GroundTruthObject,
    GroundTruthFrame,
    GroundTruthDataset,
    DataQualityReport,
)
from .metrics import (
    MetricResult,
    MetricsCalculator,
)
from .ablation import (
    AblationRecord,
    AblationStudyEngine,
)
from .error_analysis import (
    FailureCase,
    ErrorTypeSummary,
    ErrorAnalyzer,
)
from .benchmark import (
    LatencyProfile,
    ResourceProfile,
    BenchmarkRunner,
)
from .report_generator import ResearchReportGenerator
from .logger import ExperimentLogger, ExperimentEventRecord, NavigationLogger, FrameLogRecord
from .real_world_logger import RealWorldLogger, RealWorldEventRecord

__all__ = [
    "GroundTruthBBox",
    "GroundTruthObject",
    "GroundTruthFrame",
    "GroundTruthDataset",
    "DataQualityReport",
    "MetricResult",
    "MetricsCalculator",
    "AblationRecord",
    "AblationStudyEngine",
    "FailureCase",
    "ErrorTypeSummary",
    "ErrorAnalyzer",
    "LatencyProfile",
    "ResourceProfile",
    "BenchmarkRunner",
    "ResearchReportGenerator",
    "ExperimentLogger",
    "ExperimentEventRecord",
    "NavigationLogger",
    "FrameLogRecord",
    "RealWorldLogger",
    "RealWorldEventRecord",
]
