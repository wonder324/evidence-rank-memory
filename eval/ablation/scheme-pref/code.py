"""Scheme pref. Live gate: SCHEME_PREF=1 in app/extract.py."""

import os
import re

from app.extract import _add


def apply_pref_shift(facts, sentence: str) -> None:
    if os.environ.get("SCHEME_PREF", "0").strip() not in {"1", "true", "yes"}:
        return
    pref_shift = re.search(
        r"我以前喜欢([^，。！？\n]{1,40})[，,]\s*现在喜欢([^，。！？\n]{1,40})",
        sentence,
    )
    if not pref_shift:
        return
    _add(facts, "user", "preference", pref_shift.group(1), sentence, "replace")
    _add(facts, "user", "preference", pref_shift.group(2), sentence, "replace")
