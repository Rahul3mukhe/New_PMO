from typing import Dict, List, Tuple

def _line_count(text: str) -> int:
    return len([ln for ln in text.splitlines() if ln.strip()])

def _bullet_count(text: str) -> int:
    return len([ln for ln in text.splitlines() if ln.strip().startswith("- ")])

def _missing_sections(md: str, required_sections: List[str]) -> List[str]:
    low = md.lower()
    missing = []
    for s in required_sections:
        if f"## {s.lower()}" not in low:
            missing.append(s)
    return missing

def validate_doc(doc_type: str, md: str, standards: Dict) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    doc_std = standards["docs"][doc_type]
    q = standards.get("quality", {})

    if not md.strip():
        return "NOT_AVAILABLE", ["No content present"]

    low = md.lower()

    for bad in q.get("reject_if_contains", []):
        if bad in low:
            reasons.append(f"Contains disallowed placeholder/text: '{bad}'")

    min_lines = int(doc_std.get("min_total_lines", 0))
    lc = _line_count(md)
    if lc < min_lines:
        reasons.append(f"Too short: {lc} lines (<{min_lines})")

    missing = _missing_sections(md, doc_std.get("required_sections", []))
    if missing:
        reasons.append(f"Missing required sections: {', '.join(missing)}")

    for must_all in doc_std.get("must_include_all_keywords", []):
        for kw in must_all:
            if kw.lower() not in low:
                reasons.append(f"Missing required keyword: {kw}")

    for any_group in doc_std.get("must_include_any_keywords", []):
        if not any(kw.lower() in low for kw in any_group):
            reasons.append(f"Missing any of keywords: {', '.join(any_group)}")

    if "min_bullets" in doc_std:
        mb = int(doc_std["min_bullets"])
        bc = _bullet_count(md)
        if bc < mb:
            reasons.append(f"Not enough bullet items: {bc} (<{mb})")

    status = "SUFFICIENT" if not reasons else "NOT_SUFFICIENT"
    return status, reasons