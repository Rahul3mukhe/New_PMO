# llm_providers.py
# Provider cascade: Groq → Ollama (local) → Mistral API → Local Template
import os
from typing import Optional, Any

# ─────────────────────────────────────────────────────────────────────────────
# API KEYS  ←  ADD YOUR KEYS HERE
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_GROQ_KEY    = os.getenv("GROQ_API_KEY")  # ← REPLACE
  # ← ADD YOUR MISTRAL KEY (free at console.mistral.ai)

# Groq models to try in order before giving up on Groq
GROQ_MODELS_FALLBACK = [
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# Ollama model to try (must be pulled locally: `ollama pull mistral`)
OLLAMA_MODEL   = "mistral"
OLLAMA_BASE    = "http://localhost:11434"

# Mistral API model
MISTRAL_MODEL  = "mistral-small-latest"


# ─────────────────────────────────────────────────────────────────────────────
# STATUS LOG  – records which provider/model was used for each doc
# Consumed by app.py to show live progress on screen
# ─────────────────────────────────────────────────────────────────────────────
_status_log: list[dict] = []

def get_status_log() -> list[dict]:
    return list(_status_log)

def clear_status_log() -> None:
    _status_log.clear()

def _log(doc_type: str, provider: str, model: str, status: str, note: str = "") -> None:
    _status_log.append({
        "doc":      doc_type or "—",
        "provider": provider,
        "model":    model,
        "status":   status,   # "ok" | "fallback" | "failed"
        "note":     note,
    })


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _patch_ssl() -> None:
    try:
        import certifi
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        os.environ.setdefault("SSL_CERT_FILE",       certifi.where())
    except Exception:
        pass


def _resolve_key(name: str, supplied: Optional[str]) -> Optional[str]:
    if supplied and supplied.strip():
        return supplied.strip()
    env = os.getenv(f"{name.upper()}_API_KEY", "").strip()
    if env:
        return env
    if name == "groq":
        return _DEFAULT_GROQ_KEY or None
    if name == "mistral":
        return _DEFAULT_MISTRAL_KEY or None
    return None


def _is_rate_limit(err: str) -> bool:
    return "429" in err or "rate_limit" in err.lower() or "tokens per" in err.lower()


def _is_quota(err: str) -> bool:
    return "quota" in err.lower() or "billing" in err.lower() or "limit" in err.lower()


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TEMPLATE  (always works, no network required)
# ─────────────────────────────────────────────────────────────────────────────
def _local_template_generate(project, doc_type: str, standards: dict) -> str:
    if doc_type == "extraction":
        return (
            '{"project_id":"PRJ-LOCAL","project_name":"Local Fallback Project",'
            '"estimated_budget":500000.0,"actual_budget_consumed":100000.0,'
            '"total_time_taken_days":14}'
        )

    doc_std  = standards["docs"][doc_type]
    sections = doc_std["required_sections"]
    pname    = getattr(project, "project_name", "This Project")
    budget   = getattr(project, "estimated_budget", "To be confirmed")
    actual   = getattr(project, "actual_budget_consumed", "To be confirmed")

    lines = []
    for sec in sections:
        lines.append(f"## {sec}")
        sl = sec.lower()

        if sl == "purpose":
            lines += [
                f"This document defines the purpose and strategic intent of {pname}.",
                "The initiative is designed to deliver measurable business value and governance improvement.",
                "It supports the organisation's transformation objectives and compliance requirements.",
            ]
        elif sl == "scope":
            lines += [
                f"The scope of {pname} encompasses all phases from initiation through closure.",
                "In scope: requirements definition, solution design, build, test, and deployment.",
                "Out of scope: post-deployment BAU support beyond the warranty period.",
            ]
        elif "business value" in sl:
            lines += [
                "- Reduced manual effort through process automation.",
                "- Improved governance compliance and audit readiness.",
                "- Faster decision-making through standardised reporting.",
                "- Estimated ROI of 3:1 within 24 months of go-live.",
            ]
        elif "success" in sl or "metric" in sl:
            lines += [
                "- All required PMO gate documents generated and validated.",
                "- Gate pass rate of 100% across BEFORE_START, START, and END gates.",
                f"- Project delivered within approved budget of {budget}.",
                "- Stakeholder satisfaction score ≥ 4/5 in post-delivery survey.",
            ]
        elif "assumption" in sl:
            lines += [
                "- Stable requirements with no material scope changes after sign-off.",
                "- Availability of named resources throughout the project lifecycle.",
                "- Timely sign-off from stakeholders at each governance gate.",
                "- No external vendor delays beyond agreed SLAs.",
            ]
        elif "risk summary" in sl or ("risk" in sl and "detail" not in sl and "registry" not in sl and "list" not in sl):
            lines += [
                "Risks have been identified across four categories: schedule, budget, resource, and technical.",
                "All risks have been assessed for likelihood and impact.",
                "A risk owner has been assigned to each item.",
            ]
        elif "detail" in sl and "risk" in sl:
            lines += [
                "- Risk: Delayed stakeholder approvals — Likelihood: Medium, Impact: High.",
                "- Risk: Budget variance due to scope expansion — Likelihood: Medium, Impact: High.",
                "- Risk: Integration dependency failures — Likelihood: Low, Impact: High.",
                "- Risk: Resource unavailability during peak delivery phases — Likelihood: Medium, Impact: Medium.",
                "- Risk: Third-party vendor delays — Likelihood: Low, Impact: Medium.",
            ]
        elif "mitigation" in sl:
            lines += [
                "- Delayed approvals: Escalation path defined; weekly cadence with sponsor.",
                "- Budget variance: Change control process in place; contingency of 10% reserved.",
                "- Integration failures: Technical spike in Sprint 1; fallback architecture documented.",
                "- Resource unavailability: Cross-training plan in place; backup roles identified.",
                "- Vendor delays: Contractual SLA penalties; alternative supplier shortlisted.",
            ]
        elif "owner" in sl:
            lines += [
                "- Schedule risks: Project Manager",
                "- Budget risks: Finance Lead",
                "- Technical risks: Solution Architect",
                "- Resource risks: Delivery Manager",
            ]
        elif "cost estimate" in sl:
            lines += [
                f"Baseline cost estimate: {budget} (±35% confidence at this stage).",
                "Includes: infrastructure, licences, professional services, and internal resource time.",
                "Excludes: post-go-live BAU support and future enhancements.",
            ]
        elif "effort estimate" in sl:
            lines += [
                "Total estimated effort: 12 weeks, 5 FTEs.",
                "Phase 1 — Discovery & Design: 3 weeks.",
                "Phase 2 — Build & Test: 7 weeks.",
                "Phase 3 — Deployment & Handover: 2 weeks.",
            ]
        elif "confidence" in sl:
            lines += [
                "Confidence level: Medium (±35%).",
                "This estimate will be refined at the end of the Discovery phase.",
                "Key assumptions driving uncertainty are listed in the Assumptions section.",
            ]
        elif "roles" in sl or "headcount" in sl:
            lines += [
                "- Project Manager (1 FTE): Overall delivery accountability and stakeholder management.",
                "- Business Analyst (1 FTE): Requirements, process mapping, and UAT coordination.",
                "- Solution Architect (0.5 FTE): Technical design and integration oversight.",
                "- Developer (2 FTE): Build, unit test, and code review.",
                "- QA Engineer (0.5 FTE): Test strategy, execution, and defect management.",
            ]
        elif "raci" in sl:
            lines += [
                "R = Responsible, A = Accountable, C = Consulted, I = Informed",
                "- Requirements sign-off: BA (R), Sponsor (A), PMO (C), Dev (I).",
                "- Architecture decisions: Architect (R), Delivery Mgr (A), Dev (C).",
                "- Test sign-off: QA (R), PM (A), BA (C), Sponsor (I).",
                "- Go-live approval: PM (R), Sponsor (A), PMO (A), All (I).",
            ]
        elif "timeline" in sl:
            lines += [
                "Full resource coverage is planned across all project phases.",
                "Resource ramp-up begins in Week 1; peak utilisation in Weeks 4–9.",
                "Ramp-down and knowledge transfer scheduled for final two weeks.",
            ]
        elif "registry" in sl and "overview" in sl:
            lines += [
                "This registry documents all identified risks throughout the project lifecycle.",
                "It is reviewed weekly by the project team and monthly by the PMO.",
                "Risk ratings use a 5×5 likelihood-impact matrix.",
            ]
        elif "risk list" in sl:
            lines += [
                "- ID: R001 | Delayed approvals | Likelihood: 3 | Impact: 4 | Owner: PM | Status: Open",
                "- ID: R002 | Budget overrun | Likelihood: 2 | Impact: 4 | Owner: Finance | Status: Open",
                "- ID: R003 | Integration failure | Likelihood: 2 | Impact: 5 | Owner: Architect | Status: Open",
                "- ID: R004 | Key resource loss | Likelihood: 2 | Impact: 3 | Owner: Del Mgr | Status: Open",
                "- ID: R005 | Vendor SLA breach | Likelihood: 1 | Impact: 3 | Owner: PM | Status: Open",
            ]
        elif "review cadence" in sl:
            lines += [
                "Weekly: Project team risk review (standing agenda item).",
                "Monthly: PMO risk governance review.",
                "At each gate: Full risk register review and sign-off required.",
            ]
        elif "baseline budget" in sl:
            lines += [
                f"Approved baseline budget: {budget}.",
                "Budget was approved by the Investment Committee at project initiation.",
                "Includes all phases, resources, tooling, and contingency.",
            ]
        elif "actuals" in sl:
            lines += [
                f"Actual spend to date: {actual}.",
                "Spend is tracking within the approved budget envelope.",
                "Monthly finance reconciliation completed and signed off.",
            ]
        elif "variance" in sl:
            lines += [
                "Current variance: within approved ±10% tolerance.",
                "No material variances have been identified at this stage.",
                "Any future variance will be escalated via the change control process.",
            ]
        elif "forecast" in sl:
            lines += [
                "Project is forecast to complete within the approved budget.",
                "No additional funding requests are anticipated.",
                "Contingency of 10% remains available for unforeseen events.",
            ]
        elif "overview" in sl:
            lines += [
                f"{pname} is a governed initiative subject to PMO gate review.",
                "This document forms part of the mandatory governance documentation set.",
                "It must be reviewed and approved before the relevant project gate.",
            ]
        elif "approval" in sl:
            lines += [
                "| Role                | Name          | Signature | Date       |",
                "|---------------------|---------------|-----------|------------|",
                "| Project Sponsor     |               |           |            |",
                "| PMO Representative  |               |           |            |",
                "| Delivery Manager    |               |           |            |",
            ]
        else:
            lines += [
                f"{sec} details have been documented and are available for PMO review.",
                "All approvals and sign-offs are required before gate progression.",
            ]

        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _try_groq(prompt: str, model: str, api_key: Optional[str],
              temperature: float, max_tokens: int) -> str:
    _patch_ssl()
    try:
        from groq import Groq
    except ImportError as e:
        raise RuntimeError("groq not installed") from e

    key = _resolve_key("groq", api_key)
    if not key:
        raise RuntimeError("No GROQ_API_KEY available")

    kw: dict = {"api_key": key}
    try:
        import httpx
        kw["http_client"] = httpx.Client(verify=False)
    except Exception:
        pass

    client = Groq(**kw)
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": (
                "You are a senior PMO governance documentation specialist. "
                "Write formal, complete, audit-ready documents."
            )},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _try_ollama(prompt: str, model: str) -> str:
    """Call local Ollama server. Requires `ollama serve` running and model pulled."""
    import urllib.request, json as _json
    payload = _json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = _json.loads(r.read())
    text = data.get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned empty response")
    return text


def _try_mistral(prompt: str, model: str, api_key: Optional[str],
                 temperature: float, max_tokens: int) -> str:
    _patch_ssl()
    try:
        from mistralai import Mistral
    except ImportError as e:
        raise RuntimeError("mistralai not installed. Run: pip install mistralai") from e

    key = _resolve_key("mistral", api_key)
    if not key:
        raise RuntimeError("No MISTRAL_API_KEY available")

    client = Mistral(api_key=key)
    resp = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": (
                "You are a senior PMO governance documentation specialist. "
                "Write formal, complete, audit-ready documents."
            )},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def generate_text(
    provider: str,
    model: str,
    prompt: str,
    *,
    project: Optional[Any] = None,
    doc_type: Optional[str] = None,
    standards: Optional[dict] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    **kwargs,
) -> str:
    """
    Provider cascade:
      1. Groq  (tries all GROQ_MODELS_FALLBACK)
      2. Ollama (local, zero cost)
      3. Mistral API (free tier)
      4. Local Template (always works)
    """
    provider = (provider or "groq").lower().strip()

    # ── LOCAL TEMPLATE shortcut ───────────────────────────────────────────────
    if provider == "local_template":
        if project and doc_type and standards:
            _log(doc_type, "local_template", "built-in", "ok", "Offline mode")
            return _local_template_generate(project, doc_type, standards)
        return prompt

    have_project = project is not None and doc_type is not None and standards is not None

    # ── 1. GROQ ───────────────────────────────────────────────────────────────
    groq_models = (
        [model] if model not in GROQ_MODELS_FALLBACK
        else GROQ_MODELS_FALLBACK
    )
    # Always start with the configured model, then rotate through fallbacks
    if model in GROQ_MODELS_FALLBACK:
        idx = GROQ_MODELS_FALLBACK.index(model)
        groq_models = GROQ_MODELS_FALLBACK[idx:] + GROQ_MODELS_FALLBACK[:idx]
    else:
        groq_models = [model] + GROQ_MODELS_FALLBACK

    last_groq_err = ""
    for gm in groq_models:
        try:
            text = _try_groq(prompt, gm, api_key, temperature, max_tokens)
            _log(doc_type, "Groq", gm, "ok")
            return text
        except Exception as e:
            last_groq_err = str(e)
            reason = "Rate limit" if _is_rate_limit(last_groq_err) else "Error"
            _log(doc_type, "Groq", gm, "fallback", f"{reason}: {last_groq_err[:80]}")
            continue   # try next Groq model

    # ── 2. OLLAMA (local) ─────────────────────────────────────────────────────
    try:
        text = _try_ollama(prompt, OLLAMA_MODEL)
        _log(doc_type, "Ollama (local)", OLLAMA_MODEL, "ok",
             "Groq unavailable — using local Ollama")
        return text
    except Exception as e:
        _log(doc_type, "Ollama (local)", OLLAMA_MODEL, "fallback",
             f"Not available: {str(e)[:80]}")

    # ── 3. MISTRAL API ────────────────────────────────────────────────────────
    try:
        text = _try_mistral(prompt, MISTRAL_MODEL, None, temperature, max_tokens)
        _log(doc_type, "Mistral API", MISTRAL_MODEL, "ok",
             "Groq + Ollama unavailable — using Mistral")
        return text
    except Exception as e:
        _log(doc_type, "Mistral API", MISTRAL_MODEL, "fallback",
             f"Not available: {str(e)[:80]}")

    # ── 4. LOCAL TEMPLATE ─────────────────────────────────────────────────────
    if have_project:
        _log(doc_type, "Local Template", "built-in", "ok",
             "All LLM providers unavailable — using built-in template")
        return _local_template_generate(project, doc_type, standards)

    raise RuntimeError(
        f"All providers failed. Last Groq error: {last_groq_err}"
    )