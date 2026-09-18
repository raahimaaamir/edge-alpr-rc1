"""
plate_length_analysis.py

Week 9, step 2: analyzes whether Recognizer V1 shows a systematic
length/domain bias — specifically the hypothesis raised by the user's own
video 02, that the recognizer (trained entirely on 7-character Brazilian
plate formats) may tend to output 7 characters even when the true plate
is shorter.

For each (predicted_text, ground_truth_text) pair, computes:
  - ground-truth length vs. predicted length (and the raw difference)
  - exact match (after normalization)
  - insertion / deletion / substitution counts, via edit-distance with
    operation backtracking (not just the distance number)

Aggregates across a small set into: mean length difference, the fraction
of predictions matching the true length exactly, a length-difference
histogram, and the most common individual character substitutions
(feeds directly into whatever confusion-pair list is already being
tracked, e.g. Week 7's J->P finding).

This tool makes no decision by itself — per the plan, if it shows a clear
repeatable length/domain bias, that is reported as evidence for the
supervisor to weigh before any recognizer change is considered; if not,
that absence is reported too, and Recognizer V1 stays frozen either way.
"""

from dataclasses import dataclass, field
from typing import Optional

from plate_normalize import normalize_plate_text


@dataclass
class EditOps:
    insertions: int = 0    # predicted has an extra character not in ground truth
    deletions: int = 0     # ground truth has a character missing from predicted
    substitutions: int = 0  # a character present in both, but different
    matches: int = 0


def edit_distance_with_ops(pred: str, gt: str) -> EditOps:
    """Standard Levenshtein DP, but backtracks through the DP table to
    classify WHICH operations were used, not just the total distance."""
    n, m = len(pred), len(gt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # backtrack from (n, m) to (0, 0)
    ops = EditOps()
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and pred[i - 1] == gt[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.matches += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.substitutions += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.insertions += 1  # pred[i-1] has no counterpart in gt -> extra char in prediction
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.deletions += 1  # gt[j-1] has no counterpart in pred -> prediction is missing this char
            j -= 1
        else:
            break
    return ops


@dataclass
class PlateComparison:
    predicted_raw: Optional[str]
    predicted_normalized: Optional[str]
    ground_truth: str
    gt_length: int
    predicted_length: Optional[int]
    length_diff: Optional[int]  # predicted_length - gt_length
    exact_match: bool
    ops: Optional[EditOps]
    substituted_pairs: list = field(default_factory=list)  # not populated here; see analyze_set for aggregation

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["ops"] = self.ops.__dict__ if self.ops else None
        return d


def compare_one(predicted_raw: Optional[str], ground_truth: str) -> PlateComparison:
    gt_norm = normalize_plate_text(ground_truth)
    pred_norm = normalize_plate_text(predicted_raw)

    if pred_norm is None:
        return PlateComparison(
            predicted_raw=predicted_raw, predicted_normalized=None, ground_truth=gt_norm,
            gt_length=len(gt_norm), predicted_length=None, length_diff=None,
            exact_match=False, ops=None,
        )

    ops = edit_distance_with_ops(pred_norm, gt_norm)
    return PlateComparison(
        predicted_raw=predicted_raw, predicted_normalized=pred_norm, ground_truth=gt_norm,
        gt_length=len(gt_norm), predicted_length=len(pred_norm), length_diff=len(pred_norm) - len(gt_norm),
        exact_match=(pred_norm == gt_norm), ops=ops,
    )


def analyze_set(predictions_with_gt: list) -> dict:
    """predictions_with_gt: list of (predicted_text_or_None, ground_truth_text).
    Returns an aggregate report: length-bias evidence, exact-match rate,
    insertion/deletion/substitution totals, and the most common single
    character substitutions observed (aligned position by position via
    the same backtrack, extracted separately below for interpretability)."""
    comparisons = [compare_one(pred, gt) for pred, gt in predictions_with_gt]

    n = len(comparisons)
    n_with_prediction = sum(1 for c in comparisons if c.predicted_normalized is not None)
    n_exact_match = sum(1 for c in comparisons if c.exact_match)

    length_diffs = [c.length_diff for c in comparisons if c.length_diff is not None]
    length_diff_histogram = {}
    for d in length_diffs:
        length_diff_histogram[d] = length_diff_histogram.get(d, 0) + 1

    n_same_length = sum(1 for d in length_diffs if d == 0)
    n_predicted_longer = sum(1 for d in length_diffs if d > 0)
    n_predicted_shorter = sum(1 for d in length_diffs if d < 0)

    total_insertions = sum(c.ops.insertions for c in comparisons if c.ops)
    total_deletions = sum(c.ops.deletions for c in comparisons if c.ops)
    total_substitutions = sum(c.ops.substitutions for c in comparisons if c.ops)

    return {
        "n_total": n,
        "n_with_prediction": n_with_prediction,
        "n_exact_match": n_exact_match,
        "exact_match_rate": n_exact_match / n if n else None,
        "mean_length_diff": sum(length_diffs) / len(length_diffs) if length_diffs else None,
        "length_diff_histogram": length_diff_histogram,  # e.g. {0: 5, 1: 3} = 5 exact-length, 3 one-char-too-long
        "n_same_length": n_same_length,
        "n_predicted_longer": n_predicted_longer,
        "n_predicted_shorter": n_predicted_shorter,
        "total_insertions": total_insertions,
        "total_deletions": total_deletions,
        "total_substitutions": total_substitutions,
        "comparisons": [c.to_dict() for c in comparisons],
    }


def print_report(report: dict) -> None:
    print("=== Plate length / domain-bias analysis ===")
    print(f"n={report['n_total']}, with_prediction={report['n_with_prediction']}, "
          f"exact_match={report['n_exact_match']} ({report['exact_match_rate']})")
    print(f"mean length diff (predicted - ground truth): {report['mean_length_diff']}")
    print(f"length diff histogram: {report['length_diff_histogram']}")
    print(f"  same length: {report['n_same_length']}, predicted longer: {report['n_predicted_longer']}, "
          f"predicted shorter: {report['n_predicted_shorter']}")
    print(f"total insertions={report['total_insertions']}, deletions={report['total_deletions']}, "
          f"substitutions={report['total_substitutions']}")
    print()
    for c in report["comparisons"]:
        print(f"  gt={c['ground_truth']} ({c['gt_length']}) -> pred={c['predicted_normalized']} "
              f"({c['predicted_length']}), diff={c['length_diff']}, exact={c['exact_match']}, ops={c['ops']}")
