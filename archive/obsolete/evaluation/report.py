from .metrics import SystemMetrics

class ReportGenerator:
    """Generates human-readable summary reports."""
    def generate_markdown_summary(self, metrics: SystemMetrics) -> str:
        lines = [
            "# Adaptive Navigation Prototype Evaluation Summary",
            f"- **Total Frames Processed:** {metrics.total_frames}",
            f"- **Average FPS:** {metrics.average_fps:.2f}",
            f"- **Average Latency:** {metrics.average_latency_ms:.2f} ms",
            f"- **Warnings Generated:** {metrics.total_warnings_generated}",
            f"- **Warnings Suppressed (Cooldown/Priority):** {metrics.total_warnings_suppressed}",
            "",
            "### Navigation Decision Distribution"
        ]
        for decision, count in metrics.decision_distribution.items():
            lines.append(f"- **{decision}:** {count}")
        return "\n".join(lines)
