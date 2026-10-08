"""Scheme hop. Live gate: SCHEME_HOP=1 in app/store.py MemoryStore._promote_met_person_home."""

import re


def promote_met_person_home(query_text: str, hits, enabled: bool, margin: float) -> None:
    if not enabled or not hits:
        return
    if "见面" not in query_text or not re.search(r"住在|哪里", query_text):
        return
    names: list[str] = []
    for hit in hits:
        for match in re.finditer(r"和([\u4e00-\u9fffA-Za-z]{1,20})见面", hit.content):
            name = match.group(1)
            if name not in names:
                names.append(name)
    if not names:
        return
    best = max(hit.score for hit in hits)
    for hit in hits:
        text = hit.content
        if not any(name in text for name in names):
            continue
        if "residence" not in text and "住在" not in text:
            continue
        if "见面" in text and "residence" not in text:
            continue
        hit.score = max(hit.score, best + margin)
