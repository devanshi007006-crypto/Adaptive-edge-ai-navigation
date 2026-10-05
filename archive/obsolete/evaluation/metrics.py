"""
Scientific Evaluation Metrics Calculator for Step 16 Research Benchmarking.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
import math
import numpy as np


@dataclass
class MetricResult:
    name: str
    evaluated: bool
    value: Optional[float] = None
    unit: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    reason_if_not_evaluated: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class MetricsCalculator:
    """Calculates all metrics with strict validation and division-by-zero safety."""

    @staticmethod
    def calculate_detection_metrics(gt_objects_list: List[List[Any]], pred_detections_list: List[List[Any]], iou_thresh: float = 0.50) -> Dict[str, MetricResult]:
        if not gt_objects_list or all(len(gts) == 0 for gts in gt_objects_list):
            return {
                "detection_precision": MetricResult("Detection Precision", False, reason_if_not_evaluated="Not evaluated due to lack of ground truth."),
                "detection_recall": MetricResult("Detection Recall", False, reason_if_not_evaluated="Not evaluated due to lack of ground truth."),
                "detection_f1": MetricResult("Detection F1", False, reason_if_not_evaluated="Not evaluated due to lack of ground truth."),
                "mean_iou": MetricResult("Mean IoU", False, reason_if_not_evaluated="Not evaluated due to lack of ground truth."),
            }

        tp, fp, fn = 0, 0, 0
        iou_sum = 0.0

        for gts, preds in zip(gt_objects_list, pred_detections_list):
            matched_gt = set()
            for pred in preds:
                pbox = getattr(pred, "bbox", pred)
                best_iou = 0.0
                best_idx = -1
                for idx, gt in enumerate(gts):
                    if idx in matched_gt:
                        continue
                    gbox = getattr(gt, "bbox", gt)
                    ix1 = max(pbox.x1, gbox.x1)
                    iy1 = max(pbox.y1, gbox.y1)
                    ix2 = min(pbox.x2, gbox.x2)
                    iy2 = min(pbox.y2, gbox.y2)
                    iw = max(0.0, ix2 - ix1)
                    ih = max(0.0, iy2 - iy1)
                    inter = iw * ih
                    union = (pbox.width * pbox.height) + (gbox.width * gbox.height) - inter
                    iou = (inter / union) if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                if best_iou >= iou_thresh and best_idx >= 0:
                    tp += 1
                    matched_gt.add(best_idx)
                    iou_sum += best_iou
                else:
                    fp += 1

            fn += (len(gts) - len(matched_gt))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        mean_iou = (iou_sum / tp) if tp > 0 else 0.0

        return {
            "detection_precision": MetricResult("Detection Precision", True, round(prec, 4), "ratio", {"tp": tp, "fp": fp, "fn": fn}),
            "detection_recall": MetricResult("Detection Recall", True, round(rec, 4), "ratio", {"tp": tp, "fp": fp, "fn": fn}),
            "detection_f1": MetricResult("Detection F1", True, round(f1, 4), "ratio"),
            "mean_iou": MetricResult("Mean IoU", True, round(mean_iou, 4), "ratio", {"evaluated_tps": tp}),
        }

    @staticmethod
    def calculate_tracking_stability_metrics(tracked_tracks_list: List[List[Any]]) -> Dict[str, MetricResult]:
        if not tracked_tracks_list:
            return {
                "tracking_stability": MetricResult("Tracking Stability", False, reason_if_not_evaluated="No tracking data provided.")
            }

        track_durations: Dict[int, int] = {}
        for frame_tracks in tracked_tracks_list:
            for t in frame_tracks:
                tid = getattr(t, "track_id", getattr(t, "id", None))
                if tid is not None:
                    track_durations[tid] = track_durations.get(tid, 0) + 1

        if not track_durations:
            return {"tracking_stability": MetricResult("Tracking Stability", False, reason_if_not_evaluated="No valid track IDs.")}

        counts = list(track_durations.values())
        avg_lifetime = sum(counts) / len(counts)
        stable_count = sum(1 for c in counts if c >= 3)
        stability_rate = stable_count / len(counts)

        return {
            "average_track_lifetime_frames": MetricResult("Average Track Lifetime (Stability Proxy)", True, round(avg_lifetime, 2), "frames", {"total_tracks": len(counts)}),
            "track_stability_rate": MetricResult("Track Stability Rate (Stability Proxy)", True, round(stability_rate, 4), "ratio", {"stable_tracks": stable_count, "total_tracks": len(counts)}),
        }

    @staticmethod
    def calculate_depth_metrics(gt_depths: List[float], pred_depths: List[float], is_metric: bool = True) -> Dict[str, MetricResult]:
        valid_pairs = [(g, p) for g, p in zip(gt_depths, pred_depths) if g is not None and p is not None and not math.isnan(g) and not math.isnan(p)]
        if not valid_pairs:
            return {
                "depth_mae": MetricResult("Depth MAE", False, reason_if_not_evaluated="Not evaluated due to lack of ground truth depth."),
                "depth_rmse": MetricResult("Depth RMSE", False, reason_if_not_evaluated="Not evaluated due to lack of ground truth depth."),
            }

        g_arr = np.array([x[0] for x in valid_pairs], dtype=float)
        p_arr = np.array([x[1] for x in valid_pairs], dtype=float)

        if is_metric:
            errors = np.abs(g_arr - p_arr)
            mae = float(np.mean(errors))
            rmse = float(np.sqrt(np.mean((g_arr - p_arr) ** 2)))
            med_ae = float(np.median(errors))
            rel_err = float(np.mean(errors / np.maximum(g_arr, 1e-6)))

            return {
                "depth_mae": MetricResult("Depth MAE", True, round(mae, 4), "meters", {"sample_size": len(valid_pairs)}),
                "depth_rmse": MetricResult("Depth RMSE", True, round(rmse, 4), "meters"),
                "depth_median_ae": MetricResult("Depth Median Absolute Error", True, round(med_ae, 4), "meters"),
                "depth_relative_error": MetricResult("Depth Relative Error", True, round(rel_err, 4), "ratio"),
            }
        else:
            correct_pairs, total_pairs = 0, 0
            n = len(valid_pairs)
            for i in range(n):
                for j in range(i + 1, n):
                    total_pairs += 1
                    gt_rel = g_arr[i] < g_arr[j]
                    pr_rel = p_arr[i] < p_arr[j]
                    if gt_rel == pr_rel:
                        correct_pairs += 1
            ord_acc = (correct_pairs / total_pairs) if total_pairs > 0 else 0.0

            return {
                "depth_ordinal_ranking_accuracy": MetricResult("Depth Ordinal Ranking Accuracy", True, round(ord_acc, 4), "ratio", {"total_evaluated_pairs": total_pairs}),
                "depth_mae": MetricResult("Depth MAE", False, reason_if_not_evaluated="Not evaluated due to lack of metric ground truth (relative depth only)."),
            }

    @staticmethod
    def calculate_ttc_metrics(gt_ttcs: List[Optional[float]], pred_ttcs: List[Optional[float]], threshold_sec: float = 0.50) -> Dict[str, MetricResult]:
        valid_pairs = [(g, p) for g, p in zip(gt_ttcs, pred_ttcs) if g is not None and p is not None and not math.isnan(g) and not math.isnan(p)]
        if not valid_pairs:
            valid_preds = sum(1 for p in pred_ttcs if p is not None and not math.isnan(p))
            total = len(pred_ttcs) if pred_ttcs else 1
            valid_pct = valid_preds / total
            return {
                "ttc_mae": MetricResult("TTC MAE", False, reason_if_not_evaluated="Not evaluated due to lack of ground truth TTC."),
                "ttc_validity_percentage": MetricResult("TTC Validity Percentage (Proxy)", True, round(valid_pct, 4), "ratio", {"valid_count": valid_preds, "total_count": total}),
            }

        g_arr = np.array([x[0] for x in valid_pairs], dtype=float)
        p_arr = np.array([x[1] for x in valid_pairs], dtype=float)
        diffs = np.abs(g_arr - p_arr)

        mae = float(np.mean(diffs))
        rmse = float(np.sqrt(np.mean((g_arr - p_arr) ** 2)))
        med_err = float(np.median(diffs))
        within_thresh = float(np.mean(diffs <= threshold_sec))

        return {
            "ttc_mae": MetricResult("TTC MAE", True, round(mae, 4), "seconds", {"sample_size": len(valid_pairs)}),
            "ttc_rmse": MetricResult("TTC RMSE", True, round(rmse, 4), "seconds"),
            "ttc_median_error": MetricResult("TTC Median Error", True, round(med_err, 4), "seconds"),
            f"ttc_within_{threshold_sec}s_accuracy": MetricResult(f"TTC Accuracy (within ±{threshold_sec}s)", True, round(within_thresh, 4), "ratio"),
        }

    @staticmethod
    def calculate_classification_metrics(gt_labels: List[str], pred_labels: List[str], classes: List[str], metric_prefix: str = "risk") -> Dict[str, MetricResult]:
        if not gt_labels or all(g == "UNKNOWN" for g in gt_labels):
            return {
                f"{metric_prefix}_accuracy": MetricResult(f"{metric_prefix.title()} Accuracy", False, reason_if_not_evaluated=f"Not evaluated due to lack of ground truth {metric_prefix}."),
                f"{metric_prefix}_macro_f1": MetricResult(f"{metric_prefix.title()} Macro F1", False, reason_if_not_evaluated=f"Not evaluated due to lack of ground truth {metric_prefix}."),
            }

        matrix: Dict[str, Dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}
        total = 0
        correct = 0

        for g, p in zip(gt_labels, pred_labels):
            if g in matrix and p in matrix[g]:
                matrix[g][p] += 1
                total += 1
                if g == p:
                    correct += 1

        acc = (correct / total) if total > 0 else 0.0

        per_class_f1 = []
        class_details = {}
        for c in classes:
            tp = matrix[c][c]
            fp = sum(matrix[other][c] for other in classes if other != c)
            fn = sum(matrix[c][other] for other in classes if other != c)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            per_class_f1.append(f1)
            class_details[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "support": tp + fn}

        macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

        return {
            f"{metric_prefix}_accuracy": MetricResult(f"{metric_prefix.title()} Accuracy", True, round(acc, 4), "ratio", {"total_evaluated": total, "correct": correct}),
            f"{metric_prefix}_macro_f1": MetricResult(f"{metric_prefix.title()} Macro F1", True, round(macro_f1, 4), "ratio", {"per_class": class_details}),
            f"{metric_prefix}_confusion_matrix": MetricResult(f"{metric_prefix.title()} Confusion Matrix", True, None, "matrix", {"matrix": matrix, "classes": classes}),
        }

    @staticmethod
    def calculate_reliability_calibration(reliability_scores: List[float], prediction_correctness: List[bool], buckets: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0]) -> Dict[str, MetricResult]:
        if not reliability_scores or not prediction_correctness or len(reliability_scores) != len(prediction_correctness):
            return {
                "reliability_calibration": MetricResult("Reliability Calibration", False, reason_if_not_evaluated="Insufficient paired reliability/correctness samples.")
            }

        bucket_data = []
        ece_sum = 0.0
        total_samples = len(reliability_scores)

        for i in range(len(buckets) - 1):
            low, high = buckets[i], buckets[i + 1]
            indices = [idx for idx, s in enumerate(reliability_scores) if low <= s < high or (i == len(buckets) - 2 and low <= s <= high)]
            count = len(indices)
            if count > 0:
                avg_conf = float(np.mean([reliability_scores[idx] for idx in indices]))
                avg_acc = float(np.mean([1.0 if prediction_correctness[idx] else 0.0 for idx in indices]))
                err = abs(avg_conf - avg_acc)
                ece_sum += (count / total_samples) * err
            else:
                avg_conf, avg_acc, err = 0.0, 0.0, 0.0

            bucket_data.append({
                "range": f"[{low:.2f}-{high:.2f}]",
                "count": count,
                "confidence": round(avg_conf, 4),
                "accuracy": round(avg_acc, 4),
                "calibration_gap": round(err, 4),
            })

        return {
            "expected_calibration_error": MetricResult("Expected Calibration Error (ECE)", True, round(ece_sum, 4), "ratio", {"buckets": bucket_data}),
        }

    @staticmethod
    def calculate_warning_metrics(gt_warnings: List[bool], pred_warnings: List[bool]) -> Dict[str, MetricResult]:
        if not gt_warnings or len(gt_warnings) != len(pred_warnings):
            return {
                "warning_precision": MetricResult("Warning Precision", False, reason_if_not_evaluated="Not evaluated due to lack of ground truth warnings."),
                "warning_recall": MetricResult("Warning Recall", False, reason_if_not_evaluated="Not evaluated due to lack of ground truth warnings."),
            }

        tp = sum(1 for g, p in zip(gt_warnings, pred_warnings) if g and p)
        fp = sum(1 for g, p in zip(gt_warnings, pred_warnings) if not g and p)
        fn = sum(1 for g, p in zip(gt_warnings, pred_warnings) if g and not p)
        tn = sum(1 for g, p in zip(gt_warnings, pred_warnings) if not g and not p)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fwr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        mwr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return {
            "warning_precision": MetricResult("Warning Precision", True, round(prec, 4), "ratio", {"tp": tp, "fp": fp, "fn": fn, "tn": tn}),
            "warning_recall": MetricResult("Warning Recall", True, round(rec, 4), "ratio", {"tp": tp, "fp": fp, "fn": fn, "tn": tn}),
            "warning_f1": MetricResult("Warning F1", True, round(f1, 4), "ratio"),
            "false_warning_rate": MetricResult("False Warning Rate", True, round(fwr, 4), "ratio"),
            "missed_warning_rate": MetricResult("Missed Warning Rate", True, round(mwr, 4), "ratio"),
        }
