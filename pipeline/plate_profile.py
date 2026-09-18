"""
plate_profile.py

A small, configurable plate-format validator. Per the supervisor's
explicit instruction, this is NOT hard-coded UFPR logic — it's a generic
component that any jurisdiction/plate-class profile can be defined
against, and when the jurisdiction is unknown, the check is simply
unavailable (returns None, not a forced pass or fail).

A PlateProfile defines one or more valid (length -> character-class-mask)
combinations, e.g. UFPR's older Brazilian format is a single entry:
    {7: "LLLDDDD"}   # 3 letters, then 4 digits
A jurisdiction with multiple legitimate formats (e.g. old + Mercosur
layouts) would simply have multiple entries in `formats`.

validate() checks structural validity only — length and per-position
character class (L=letter, D=digit, A=either). It NEVER pads, truncates,
or substitutes to force a match; an invalid string is reported as invalid
and nothing more.

rerank_with_profile() is the OPTIONAL reranking capability the supervisor
described: given the model's own top-k alternatives at each decoder slot
(the same structure build_track0080_diagnostic_package.py's
raw_topk_probs() produces), it may substitute a character ONLY if (a) the
current candidate is otherwise valid except at that one position, (b)
the model's own alternatives at that position include one matching the
required character class, and (c) that alternative clears a minimum
credibility threshold. If no such credible, class-matching alternative
exists, it returns None (no rerank) rather than forcing anything — this
is the exact behavior that keeps AP1382 from ever becoming APJ3829: J is
not a credible model-supported alternative at that position (see the
track 0080 diagnostic — J appears in the top-3 only once, at 1.5%
probability, and is absent from the other two frames' top-3 entirely).

SCOPING NOTE: validate() is fully wired into the decision-rule comparison
(run_acceptance_rule_comparison.py, rules E/F) using the text the pipeline
already decodes. rerank_with_profile() is implemented and tested here as
a standalone capability operating on per-slot probability data (the same
structure the diagnostic script produces) — it is NOT currently wired
into the live per-track video pipeline, because doing so would require
threading full per-slot probability arrays through the tracker/
observation layer, which does not currently persist them (only the
winning character's own confidence is kept per observation). Per the
supervisor's own instruction to keep this a small, minimal component, that
larger plumbing change is not made here; reranking is available for
offline/diagnostic use and can be wired into the live path in a later,
separately-reviewed change if the accept/reject-only comparison shows it's
needed.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlateProfile:
    name: str
    formats: dict  # {length: mask_string}, mask chars: 'L' (letter), 'D' (digit), 'A' (either)


# UFPR-ALPR's documented format (see track0080_findings_memo.md): 3 letters
# + 4 digits, ranging AAA-0001 to BEZ-9999 in this dataset specifically.
UFPR_PROFILE = PlateProfile(name="UFPR (older Brazilian)", formats={7: "LLLDDDD"})


def _char_matches_class(ch: str, mask_char: str) -> bool:
    if mask_char == "A":
        return ch.isalnum()
    if mask_char == "L":
        return ch.isalpha()
    if mask_char == "D":
        return ch.isdigit()
    raise ValueError(f"unknown mask character: {mask_char!r}")


def validate(text: str, profile: Optional[PlateProfile]) -> tuple:
    """Returns (is_valid: bool, reason: Optional[str]). If profile is
    None (jurisdiction/plate type unknown), returns (None, "no_profile")
    — the check is simply unavailable, never a forced pass or fail."""
    if profile is None:
        return None, "no_profile"
    if text is None:
        return False, "no_text"

    mask = profile.formats.get(len(text))
    if mask is None:
        return False, f"invalid_length ({len(text)}, expected one of {sorted(profile.formats.keys())})"

    for i, (ch, mask_char) in enumerate(zip(text, mask)):
        if not _char_matches_class(ch, mask_char):
            return False, f"invalid_character_class at position {i} ({ch!r} does not match {mask_char!r})"

    return True, None


def rerank_with_profile(text: str, profile: Optional[PlateProfile],
                         per_slot_top_k: list, min_credible_prob: float = 0.05) -> Optional[str]:
    """Attempts to repair a SINGLE character-class violation using the
    model's own reported alternatives at that position — never by padding,
    truncating, or introducing a character the model didn't itself report
    as plausible.

    per_slot_top_k: the same structure raw_topk_probs() produces —
    [{"slot": i, "top_k": [(char, prob), ...]}, ...], one entry per
    decoder slot, in the same left-to-right order as `text`'s characters
    (pad-character slots at the end are ignored).

    Returns the repaired string if exactly one position is invalid and a
    class-matching alternative clears min_credible_prob; otherwise None
    (no rerank — including when the length itself is wrong, which this
    function deliberately never attempts to fix, matching the
    supervisor's instruction never to pad or truncate)."""
    if profile is None or text is None:
        return None

    mask = profile.formats.get(len(text))
    if mask is None:
        return None  # wrong length — not this function's job to fix

    violations = [i for i, (ch, mask_char) in enumerate(zip(text, mask)) if not _char_matches_class(ch, mask_char)]
    if len(violations) != 1:
        return None  # only handle exactly one bad position; anything else is not a simple rerank case

    pos = violations[0]
    if pos >= len(per_slot_top_k):
        return None
    required_class = mask[pos]

    for char, prob in per_slot_top_k[pos]["top_k"]:
        if char == "_":
            continue
        if _char_matches_class(char, required_class) and prob >= min_credible_prob:
            return text[:pos] + char + text[pos + 1:]

    return None  # no credible, class-matching alternative — no rerank, per instruction
