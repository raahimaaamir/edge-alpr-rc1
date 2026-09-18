# Track 0080 — Diagnostic Findings

Strictly factual summary of the accompanying diagnostic package. No
correction or architectural recommendation is proposed here — this is
evidence only, per your instruction.

**Ground truth:** `APJ3829`. **Accepted result:** `AP1382` (3/3 agreement
among the Top-3 selected observations; 15/15 usable observations across
the whole track never produced the literal correct string).

## Jurisdiction

UFPR-ALPR (Federal University of Paraná, Brazil, collected in Paraná).
Documented format: 3 letters + 4 digits (LLL-DDDD), with plates in this
dataset ranging from AAA-0001 to BEZ-9999. `APJ3829` matches this format
exactly, and its leading letter falls inside the documented range. (Source:
dataset paper and a follow-up analysis of it — see prior message for
citations.)

## Finding 1 — the disputed character (position 3, ground truth "J")

Across all 15 usable observations, the decoded character at this position
was: `1` (9 times), `U` (4 times), and `I`-adjacent readings twice
(`API382`, `AP13829`). **`J` was never the decoded output on any
observation.**

Raw per-position top-3 probabilities (from the ONNX model directly, not
derived) for the three Top-3 observations:

| Frame | 1st | 2nd | 3rd |
|---|---|---|---|
| 27 (agreement_count vote, highest confidence) | `1` (93.4%) | `U` (2.2%) | `J` (1.5%) |
| 1 | `1` (76.6%) | `U` (8.5%) | `I` (6.7%) — `J` not in top-3 |
| 3 | `1` (74.5%) | `I` (9.7%) | `U` (7.5%) — `J` not in top-3 |

`J` appears as a model-reported alternative in only one of the three
frames, and there only in third place at 1.5% probability. In the other
two frames it does not appear in the top-3 at all. The model's confusion
at this position is consistently within a `1`/`U`/`I` family, not a
`1`/`J` pair.

## Finding 2 — separate ambiguity at the final character position

At decoder slot 6 (the 7th/last character), all three Top-3 observations
show a consistent split between the pad/end-of-plate token and the digit
`9`:

| Frame | Pad token | `9` |
|---|---|---|
| 27 | 55.7% | 35.4% |
| 1 | 77.3% | 11.2% |
| 3 | 66.2% | 19.5% |

This is a distinct issue from Finding 1 — a length-boundary ambiguity at
the crop's trailing edge, rather than a character-identity confusion. The
uploaded original-resolution crops show the plate's right edge sitting
close to the crop boundary in several frames.

## Supporting data

- `summary.json` — full per-frame detail (bbox, crop dimensions, all
  confidences, complete top-3-per-slot probabilities) for the three
  Top-3 observations.
- `all_observations.csv` — the full 15-observation vote/confidence table.
- 6 crop images: original-resolution and preprocessed (as fed to
  Recognizer V1.1) for each of the three Top-3 frames.
