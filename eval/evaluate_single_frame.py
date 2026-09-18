"""
evaluate_single_frame.py

The one standardized evaluation function for the single-frame ALPR
pipeline, frozen as of Week 5. Every later version of the pipeline
(fine-tuned detector, retrained recognizer, new quality gates, etc.) is
scored by calling evaluate_single_frame() unchanged, so numbers are
comparable release over release rather than each version inventing its
own metric definitions.

Metrics computed (per the Week 5 spec):
  - exact plate accuracy      — status=="ok" AND plate_text == ground truth,
                                 as a fraction of ALL samples (not just the
                                 ones the pipeline chose to answer). A missed
                                 detection counts as a failure here, same as
                                 a wrong OCR read — this is an end-to-end
                                 number, not a recognizer-only number.
  - character accuracy / CER  — Levenshtein-based. Reported TWO ways because
                                 they answer different questions:
                                   * cer_on_accepted:  only over "ok" samples
                                     with a predicted string — "how good is
                                     the recognizer when it commits to an
                                     answer?"
                                   * cer_end_to_end: every sample counted,
                                     with a missed/errored sample scored as
                                     if it had predicted an empty string
                                     (i.e. maximally wrong) — "how good is
                                     the whole pipeline, including misses?"
  - coverage                  — fraction of samples where status=="ok"
                                 (the pipeline was willing to commit to an
                                 answer at all)
  - incorrect accepted results— "ok" samples where plate_text != ground
                                 truth: a CONFIDENT WRONG answer, which is a
                                 worse failure mode than abstaining. Reported
                                 both as a share of all samples and as a
                                 share of accepted (covered) samples.
  - latency                   — mean/median/p95 total_ms, plus mean per-stage
                                 ms (detect/crop/quality/recognize), pulled
                                 straight from each result's StageTimings.

Ground truth matching is exact string equality (case-sensitive), consistent
with the convention already used for Recognizer V1's freeze report.
"""

from dataclasses import dataclass, field
from typing import Optional


def _levenshtein(a: str, b: str) -> int:
    """Standard edit-distance DP. Plate strings are short (<=10 chars), so
    no need for anything fancier."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost # substitution
            )
        prev = curr
    return prev[-1]


@dataclass
class LatencyStats:
    mean_ms: Optional[float] = None
    median_ms: Optional[float] = None
    p95_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {"mean_ms": self.mean_ms, "median_ms": self.median_ms, "p95_ms": self.p95_ms}


@dataclass
class EvalReport:
    n_total: int = 0
    n_ok: int = 0
    n_correct: int = 0
    n_incorrect_accepted: int = 0
    n_not_covered: int = 0  # no_detection / low_quality / error

    exact_accuracy: Optional[float] = None
    coverage: Optional[float] = None
    incorrect_accepted_rate_of_total: Optional[float] = None
    incorrect_accepted_rate_of_accepted: Optional[float] = None

    char_accuracy_on_accepted: Optional[float] = None
    cer_on_accepted: Optional[float] = None
    char_accuracy_end_to_end: Optional[float] = None
    cer_end_to_end: Optional[float] = None

    latency_total: LatencyStats = field(default_factory=LatencyStats)
    latency_detect_mean_ms: Optional[float] = None
    latency_crop_mean_ms: Optional[float] = None
    latency_quality_mean_ms: Optional[float] = None
    latency_recognize_mean_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_ok": self.n_ok,
            "n_correct": self.n_correct,
            "n_incorrect_accepted": self.n_incorrect_accepted,
            "n_not_covered": self.n_not_covered,
            "exact_accuracy": self.exact_accuracy,
            "coverage": self.coverage,
            "incorrect_accepted_rate_of_total": self.incorrect_accepted_rate_of_total,
            "incorrect_accepted_rate_of_accepted": self.incorrect_accepted_rate_of_accepted,
            "char_accuracy_on_accepted": self.char_accuracy_on_accepted,
            "cer_on_accepted": self.cer_on_accepted,
            "char_accuracy_end_to_end": self.char_accuracy_end_to_end,
            "cer_end_to_end": self.cer_end_to_end,
            "latency_total": self.latency_total.to_dict(),
            "latency_detect_mean_ms": self.latency_detect_mean_ms,
            "latency_crop_mean_ms": self.latency_crop_mean_ms,
            "latency_quality_mean_ms": self.latency_quality_mean_ms,
            "latency_recognize_mean_ms": self.latency_recognize_mean_ms,
        }


def _percentile(values: list, p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def evaluate_single_frame(results: list, ground_truths: list) -> EvalReport:
    """
    results: list of PlateResult (or anything duck-typed the same way —
        .status, .plate_text, .timings.{detect_ms,crop_ms,quality_ms,
        recognize_ms,total_ms})
    ground_truths: list of ground-truth plate text strings, same length and
        order as results (one per sample/frame)
    """
    if len(results) != len(ground_truths):
        raise ValueError(f"results ({len(results)}) and ground_truths ({len(ground_truths)}) must be the same length")

    report = EvalReport()
    report.n_total = len(results)

    accepted_edit_distances = []
    accepted_gt_lengths = []
    end_to_end_edit_distances = []
    end_to_end_gt_lengths = []

    total_ms_list = []
    detect_ms_list = []
    crop_ms_list = []
    quality_ms_list = []
    recognize_ms_list = []

    for result, gt in zip(results, ground_truths):
        is_ok = getattr(result, "status", None) == "ok"
        pred = getattr(result, "plate_text", None) if is_ok else None

        if is_ok:
            report.n_ok += 1
            if pred == gt:
                report.n_correct += 1
            else:
                report.n_incorrect_accepted += 1

            dist = _levenshtein(pred or "", gt)
            accepted_edit_distances.append(dist)
            accepted_gt_lengths.append(len(gt))

            end_to_end_edit_distances.append(dist)
            end_to_end_gt_lengths.append(len(gt))
        else:
            report.n_not_covered += 1
            # end-to-end CER treats a non-"ok" sample as a fully-wrong
            # (empty-string) prediction, per the docstring above
            end_to_end_edit_distances.append(len(gt))
            end_to_end_gt_lengths.append(len(gt))

        timings = getattr(result, "timings", None)
        if timings is not None:
            total_ms_list.append(getattr(timings, "total_ms", 0.0))
            detect_ms_list.append(getattr(timings, "detect_ms", 0.0))
            crop_ms_list.append(getattr(timings, "crop_ms", 0.0))
            quality_ms_list.append(getattr(timings, "quality_ms", 0.0))
            recognize_ms_list.append(getattr(timings, "recognize_ms", 0.0))

    report.exact_accuracy = report.n_correct / report.n_total if report.n_total else None
    report.coverage = report.n_ok / report.n_total if report.n_total else None
    report.incorrect_accepted_rate_of_total = (
        report.n_incorrect_accepted / report.n_total if report.n_total else None
    )
    report.incorrect_accepted_rate_of_accepted = (
        report.n_incorrect_accepted / report.n_ok if report.n_ok else None
    )

    if accepted_gt_lengths and sum(accepted_gt_lengths) > 0:
        cer_acc = sum(accepted_edit_distances) / sum(accepted_gt_lengths)
        report.cer_on_accepted = cer_acc
        report.char_accuracy_on_accepted = max(0.0, 1.0 - cer_acc)

    if end_to_end_gt_lengths and sum(end_to_end_gt_lengths) > 0:
        cer_e2e = sum(end_to_end_edit_distances) / sum(end_to_end_gt_lengths)
        report.cer_end_to_end = cer_e2e
        report.char_accuracy_end_to_end = max(0.0, 1.0 - cer_e2e)

    if total_ms_list:
        report.latency_total = LatencyStats(
            mean_ms=sum(total_ms_list) / len(total_ms_list),
            median_ms=_percentile(total_ms_list, 0.5),
            p95_ms=_percentile(total_ms_list, 0.95),
        )
    if detect_ms_list:
        report.latency_detect_mean_ms = sum(detect_ms_list) / len(detect_ms_list)
    if crop_ms_list:
        report.latency_crop_mean_ms = sum(crop_ms_list) / len(crop_ms_list)
    if quality_ms_list:
        report.latency_quality_mean_ms = sum(quality_ms_list) / len(quality_ms_list)
    if recognize_ms_list:
        report.latency_recognize_mean_ms = sum(recognize_ms_list) / len(recognize_ms_list)

    return report


def print_report(report: EvalReport) -> None:
    d = report.to_dict()
    print("=== Single-frame pipeline evaluation ===")
    print(f"n_total={d['n_total']}  n_ok={d['n_ok']}  n_correct={d['n_correct']}  "
          f"n_incorrect_accepted={d['n_incorrect_accepted']}  n_not_covered={d['n_not_covered']}")
    print(f"Exact plate accuracy (end-to-end): {d['exact_accuracy']}")
    print(f"Coverage: {d['coverage']}")
    print(f"Incorrect-accepted rate (of total / of accepted): "
          f"{d['incorrect_accepted_rate_of_total']} / {d['incorrect_accepted_rate_of_accepted']}")
    print(f"Char accuracy on accepted (CER): {d['char_accuracy_on_accepted']} ({d['cer_on_accepted']})")
    print(f"Char accuracy end-to-end (CER): {d['char_accuracy_end_to_end']} ({d['cer_end_to_end']})")
    print(f"Latency total ms — mean/median/p95: "
          f"{d['latency_total']['mean_ms']}/{d['latency_total']['median_ms']}/{d['latency_total']['p95_ms']}")
    print(f"Latency per stage (mean ms) — detect={d['latency_detect_mean_ms']}, "
          f"crop={d['latency_crop_mean_ms']}, quality={d['latency_quality_mean_ms']}, "
          f"recognize={d['latency_recognize_mean_ms']}")
