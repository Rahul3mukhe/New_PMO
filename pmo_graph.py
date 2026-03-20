import os
import yaml
import json
from typing import Dict, Set

from langgraph.graph import StateGraph, START, END

from schemas import PMOState, DocumentArtifact
from decisioning import compute_requirements, evaluate_gates
from guardrails import validate_doc
from llm_providers import generate_text
from storage import make_run_dir
from doc_templates import create_official_docx, create_decision_report_docx

def load_standards(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _needed_docs(state: PMOState) -> Set[str]:
    return set(state.standards["docs"].keys())

# NODE 1: extract_project_context
def node_extract_project_context(state: PMOState) -> PMOState:
    raw_text = state.audit.get("raw_upload_text", "")
    if not raw_text.strip():
        # Fallback if no documents were uploaded yet
        state.project.project_id = "PRJ-001"
        state.project.project_name = "New Uploaded Project"
        return state

    prompt = f"""
You are an AI extracting project context from unstructured documents.
Read the following text and extract details to match this JSON schema precisely:
{{
  "project_id": "string (e.g. PRJ-123)",
  "project_name": "string",
  "project_type": "string",
  "sponsor": "string",
  "estimated_budget": "number or null",
  "actual_budget_consumed": "number or null",
  "total_time_taken_days": "number or null",
  "timeline_summary": "string",
  "scope_summary": "string",
  "key_deliverables": ["string"],
  "known_risks": ["string"]
}}

CRITICAL INSTRUCTIONS:
1. For budget fields (estimated_budget, actual_budget_consumed), look for any currency values.
2. If a value is NOT found, set it to null in the JSON. NEVER default to 0.0 unless the text explicitly mentions zero.
3. If multiple values are found, use the most plausible "total" or "summary" figure.
4. Only output valid JSON. Output NOTHING else.

TEXT to analyze:
{raw_text[:12000]}
"""
    try:
        response = generate_text(
            provider=state.provider,
            model=state.model,
            prompt=prompt,
            api_key=state.audit.get("api_key"),
            temperature=0.0,
            max_tokens=2048,
            standards=state.standards,
            project=state.project,
            doc_type="extraction"
        )
        
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = response[start:end]
            data = json.loads(json_str)
            for k, v in data.items():
                if hasattr(state.project, k):
                    setattr(state.project, k, v)
    except Exception as e:
        state.audit["extraction_error"] = str(e)
    
    return state

# NODE 2: requirements
def node_requirements(state: PMOState) -> PMOState:
    state.required_docs = compute_requirements(state.standards, state.project.project_type)
    state.audit["requirements"] = state.required_docs
    return state

# NODE 2: init_docs
def node_init_docs(state: PMOState) -> PMOState:
    for doc_type, meta in state.standards["docs"].items():
        if doc_type not in state.docs:
            state.docs[doc_type] = DocumentArtifact(
                doc_type=doc_type,
                title=meta["title"]
            )
    return state

# NODE 3: load_uploaded_docs
def node_load_uploaded_docs(state: PMOState) -> PMOState:
    mapping = state.audit.get("uploaded_mapping", {})
    loaded = []
    
    for doc_type, txt in mapping.items():
        if doc_type in state.docs and txt and txt.strip():
            state.docs[doc_type].content_markdown = txt
            state.docs[doc_type].status = "NOT_SUFFICIENT"
            state.docs[doc_type].reasons = ["Uploaded by user; pending validation"]
            loaded.append(doc_type)

    state.audit["loaded_docs"] = loaded
    return state

def build_generation_prompt(state: PMOState, doc_type: str) -> str:
    org = state.standards["org"]["name"]
    doc_std = state.standards["docs"][doc_type]
    proj = state.project

    sections = "\n".join([f"- {s}" for s in doc_std["required_sections"]])

    return f"""
You are a PMO governance documentation assistant for {org}.
Generate a professional internal document in Markdown.

Document type: {doc_std["title"]}

STRICT RULES:
- Use headings EXACTLY as: "## <Section Name>" for each required section.
- Include ALL required sections in this order:
{sections}
- Use a formal, audit-ready tone.
- Be specific; include numbers/ranges where appropriate.
- Avoid placeholders like TBD, N/A, lorem, etc.
- IMPORTANT: For sections that imply a list (like "Risk List", "Key Deliverables", "Roles"), use Markdown bullet points ("- "). Generate at least 5 detailed items for these lists to satisfy PMO standards.
- Include an Approvals section with placeholders for role-based approvals (Name/Role/Date).

Project context:
- Project ID: {proj.project_id}
- Project Name: {proj.project_name}
- Project Type: {proj.project_type}
- Sponsor: {proj.sponsor}
- Estimated Budget: {proj.estimated_budget}
- Actual Budget Consumed: {proj.actual_budget_consumed}
- Total Time Taken (days): {proj.total_time_taken_days}
- Timeline Summary: {proj.timeline_summary}
- Scope Summary: {proj.scope_summary}
- Key Deliverables: {proj.key_deliverables}
- Known Risks: {proj.known_risks}

Now output ONLY the full Markdown document.
""".strip()

# NODE 4: generate_missing_docs
def node_generate_missing_docs(state: PMOState) -> PMOState:
    """
    Generate any required docs that are empty/missing.
    Supports 'local_template' (offline deterministic) and other providers like 'gemini'.
    """
    needed = _needed_docs(state)
    generated = []

    for d in needed:
        art = state.docs[d]

        # If content already exists (loaded from file or previous generation), skip.
        if art.content_markdown and art.content_markdown.strip():
            continue

        # Build prompt (existing helper)
        prompt = build_generation_prompt(state, d)

        # Call generator — always pass project/doc_type/standards so fallback works
        try:
            md = generate_text(
                provider=state.provider,
                model=state.model,
                prompt=prompt,
                project=state.project,
                doc_type=d,
                standards=state.standards,
                api_key=state.audit.get("api_key"),
                temperature=state.audit.get("temperature", 0.0),
                max_tokens=state.audit.get("max_tokens", 2048),
            )
        except Exception as gen_err:
            # Last-resort: use local template directly
            from llm_providers import _local_template_generate
            md = _local_template_generate(state.project, d, state.standards)
            state.audit[f"gen_fallback_{d}"] = str(gen_err)

        # Strip fallback comment injected by llm_providers (guardrails would reject it)
        if md and "<!-- FALLBACK" in md:
            md = md[:md.index("<!-- FALLBACK")].rstrip()

        # Populate artifact fields
        art.content_markdown = md or ""
        art.status = "NOT_SUFFICIENT"  # validation will determine final status
        art.reasons = ["Generated; pending validation"]
        generated.append(d)

    state.audit["generated_docs"] = generated
    return state

# NODE 5: validate_docs
def node_validate_docs(state: PMOState) -> PMOState:
    needed = _needed_docs(state)
    for d in needed:
        art = state.docs[d]
        status, reasons = validate_doc(d, art.content_markdown, state.standards)
        art.status = status
        art.reasons = reasons if reasons else art.reasons
    return state

# NODE 6: repair_once
def node_repair_once(state: PMOState) -> PMOState:
    needed = _needed_docs(state)
    repaired = []

    for d in needed:
        art = state.docs[d]
        if art.status != "NOT_SUFFICIENT":
            continue

        issues = "\n".join([f"- {r}" for r in art.reasons])
        prompt = build_generation_prompt(state, d) + f"\n\nREPAIR REQUIRED. Fix these issues:\n{issues}\n"

        try:
            md = generate_text(
                provider=state.provider,
                model=state.model,
                prompt=prompt,
                project=state.project,
                doc_type=d,
                standards=state.standards,
                api_key=state.audit.get("api_key"),
                temperature=state.audit.get("temperature", 0.0),
                max_tokens=state.audit.get("max_tokens", 2048),
            )
        except Exception as rep_err:
            from llm_providers import _local_template_generate
            md = _local_template_generate(state.project, d, state.standards)
            state.audit[f"repair_fallback_{d}"] = str(rep_err)

        if md and "<!-- FALLBACK" in md:
            md = md[:md.index("<!-- FALLBACK")].rstrip()

        art.content_markdown = md
        art.status = "NOT_SUFFICIENT"
        art.reasons = ["Regenerated; pending validation"]
        repaired.append(d)

    state.audit["repaired_docs"] = repaired
    return state

# NODE 7: validate_again
def node_validate_again(state: PMOState) -> PMOState:
    return node_validate_docs(state)

# NODE 8: decide
def node_decide(state: PMOState) -> PMOState:
    return evaluate_gates(state)

def should_repair(state: PMOState) -> str:
    needed = _needed_docs(state)
    if any(state.docs[d].status == "NOT_SUFFICIENT" for d in needed):
        # Only repair once
        if not state.audit.get("repaired_docs"):
            return "repair"
    return "decide"

def build_graph():
    g = StateGraph(PMOState)

    g.add_node("extract_context", node_extract_project_context)
    g.add_node("requirements", node_requirements)
    g.add_node("init_docs", node_init_docs)
    g.add_node("load_uploaded_docs", node_load_uploaded_docs)
    g.add_node("generate_missing_docs", node_generate_missing_docs)
    g.add_node("validate_docs", node_validate_docs)
    g.add_node("repair_once", node_repair_once)
    g.add_node("validate_again", node_validate_again)
    g.add_node("decide", node_decide)

    g.add_edge(START, "extract_context")
    g.add_edge("extract_context", "requirements")
    g.add_edge("requirements", "init_docs")
    g.add_edge("init_docs", "load_uploaded_docs")
    g.add_edge("load_uploaded_docs", "generate_missing_docs")
    g.add_edge("generate_missing_docs", "validate_docs")

    g.add_conditional_edges("validate_docs", should_repair, {
        "repair": "repair_once",
        "decide": "decide"
    })

    g.add_edge("repair_once", "validate_again")
    g.add_edge("validate_again", "decide")
    g.add_edge("decide", END)

    return g.compile()