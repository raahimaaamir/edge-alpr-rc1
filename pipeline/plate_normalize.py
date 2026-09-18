"""
plate_normalize.py

The one normalization convention for comparing predicted plate text against
ground truth, used consistently from Week 6 onward per the supervisor's
instruction: uppercase, strip permitted spacing/hyphen formatting. Raw
(unmodified) predictions are always retained separately alongside the
normalized form — normalization is applied only at comparison time, never
by discarding the original OCR output.

This is currently close to a no-op for Recognizer V1 specifically (its
output charset "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_" already has no
spaces/hyphens and is already uppercase — see recognizer_v1_spec.md), but
the function exists so the convention is explicit, tested, and applied
identically regardless of which recognizer version produced a given
prediction.
"""

import re
from typing import Optional

# Characters treated as "permitted formatting" — stripped before comparison.
# Space and hyphen are the two conventional plate-formatting separators
# (e.g. "ABC-1234" or "ABC 1234"); anything else is left alone so a genuine
# OCR error isn't silently erased by normalization.
_FORMATTING_CHARS_RE = re.compile(r"[\s\-]")


def normalize_plate_text(text: Optional[str]) -> Optional[str]:
    """Uppercase + strip permitted spacing/hyphen formatting. Returns None
    unchanged (so callers don't need a None-check before calling this)."""
    if text is None:
        return None
    return _FORMATTING_CHARS_RE.sub("", text).upper()
