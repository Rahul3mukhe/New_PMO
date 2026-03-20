from typing import Dict, List
from schemas import PMOState, GateResult

def compute_requirements(standards: Dict, project_type: str) -> Dict[str, List[str]]:
    types = standards["project_types"]
    if project_type not in types:
        project_type = "Default"

    spec = dict(types[project_type])
    base = {}
    if "inherits" in spec:
        base = dict(types[spec["inherits"]])

    def merged(key: str) -> List[str]:
        return list(dict.fromkeys((base.get(key, []) + spec.get(key, []))))

    return {
        "BEFORE_START": merged("required_docs_before_start"),
        "START": merged("required_docs_start_gate"),
        "BEFORE_END": merged("required_before_end_metrics"),  # metrics keys
        "END": merged("required_docs_end_gate"),
    }

def evaluate_gates(state: PMOState) -> PMOState:
    req = state.required_docs
    docs = state.docs
    proj = state.project

    gates: List[GateResult] = []

    # BEFORE_START docs
    findings = []
    passed = True
    for d in req["BEFORE_START"]:
        if docs[d].status != "SUFFICIENT":
            passed = False
            findings.append(f"{d}: {docs[d].status} ({'; '.join(docs[d].reasons)})")
    gates.append(GateResult(gate="BEFORE_START", passed=passed, findings=findings))

    # START docs
    findings = []
    passed = True
    for d in req["START"]:
        if docs[d].status != "SUFFICIENT":
            passed = False
            findings.append(f"{d}: {docs[d].status} ({'; '.join(docs[d].reasons)})")
    gates.append(GateResult(gate="START", passed=passed, findings=findings))

    # BEFORE_END metrics
    findings = []
    passed = True
    for m in req["BEFORE_END"]:
        if getattr(proj, m, None) is None:
            passed = False
            findings.append(f"Missing metric: {m}")
    gates.append(GateResult(gate="BEFORE_END", passed=passed, findings=findings))

    # END docs
    findings = []
    passed = True
    for d in req["END"]:
        if docs[d].status != "SUFFICIENT":
            passed = False
            findings.append(f"{d}: {docs[d].status} ({'; '.join(docs[d].reasons)})")
    gates.append(GateResult(gate="END", passed=passed, findings=findings))

    state.gates = gates
    overall = all(g.passed for g in gates)
    state.decision = "APPROVE" if overall else "INVALIDATE"

    if overall:
        state.summary = "All gates passed."
    else:
        failed = [g for g in gates if not g.passed]
        state.summary = " | ".join(
            [f"{g.gate} failed: {', '.join(g.findings) if g.findings else 'issues'}" for g in failed]
        )

    return state