"""Navigation and spatial reasoning package."""
from .spatial import SpatialObjectRepresentation, SpatialZoneTracker, SpatialAnalyzer
from .path_geometry import PathOverlapAssessment, PathGeometryAnalyzer
from .navigation_decision import (
    NavigationDecision,
    SceneNavigationState,
    NavigationEngine,
)

# Preserve legacy symbols for backward compatibility
from .zones import ZoneType, ZoneRisk, ZoneDivider
from .free_space import FreeSpaceSegment, FreeSpaceAnalyzer

__all__ = [
    "SpatialObjectRepresentation",
    "SpatialZoneTracker",
    "SpatialAnalyzer",
    "PathOverlapAssessment",
    "PathGeometryAnalyzer",
    "NavigationDecision",
    "SceneNavigationState",
    "NavigationEngine",
    "ZoneType",
    "ZoneRisk",
    "ZoneDivider",
    "FreeSpaceSegment",
    "FreeSpaceAnalyzer",
]
