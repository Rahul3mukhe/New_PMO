# app.py — PMO Agent
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import os, io, json, zipfile
import streamlit as st
from datetime import datetime
from pathlib import Path

from pmo_graph import build_graph, load_standards
from schemas import Project, PMOState

# ── CONFIG ────────────────────────────────────────────────────────────────────
STANDARDS_PATH  = "config/standards.yml"
PROVIDER        = "groq"
MODEL           = "llama-3.3-70b-versatile"
MAX_PREVIEW     = 300
DOC_TYPES       = [
    "proof_of_value", "risk_assessment", "rough_order_estimation",
    "team_sizing", "risk_registry", "budget_statement",
]
DOC_INFO = {
    "proof_of_value":         {"title": "Proof of Value",               "gate": "Before Start", "desc": "Business justification, expected value, scope and success metrics.", "sections": ["Purpose","Scope","Business Value","Success Metrics","Assumptions","Approvals"]},
    "risk_assessment":        {"title": "Risk Assessment",              "gate": "Before Start", "desc": "Identifies risks, evaluates likelihood and impact, defines mitigations.", "sections": ["Overview","Risk Summary","Detailed Risks","Mitigations","Owners","Approvals"]},
    "rough_order_estimation": {"title": "Rough Order of Estimation",    "gate": "Before Start", "desc": "High-level cost and effort estimate (±35% confidence) with stated assumptions.", "sections": ["Overview","Cost Estimate","Effort Estimate","Assumptions","Confidence Level","Approvals"]},
    "team_sizing":            {"title": "Team Sizing",                  "gate": "Before Start", "desc": "Roles, headcount, RACI and resource coverage required to deliver.", "sections": ["Overview","Roles and Headcount","RACI","Timeline Coverage","Approvals"]},
    "risk_registry":          {"title": "Risk Registry",               "gate": "Start Gate",   "desc": "Live register of all identified risks with owners, ratings and review cadence.", "sections": ["Registry Overview","Risk List","Review Cadence","Approvals"]},
    "budget_statement":       {"title": "Budget Statement",             "gate": "End Gate",     "desc": "Formal comparison of approved budget vs actual spend with variance explanation.", "sections": ["Overview","Baseline Budget","Actuals","Variance Explanation","Forecast","Approvals"]},
}
SAMPLE_PROJECT_JSON = '''{
  "project_id": "PRJ-001",
  "project_name": "PMO Agent Professional MVP",
  "project_type": "Regulated",
  "sponsor": "VP Delivery",
  "estimated_budget": 200000,
  "actual_budget_consumed": 135000,
  "total_time_taken_days": 192,
  "timeline_summary": "Planned 12 weeks; delivered in 13 weeks including UAT stabilization.",
  "scope_summary": "Automate PMO gate checks and generate governance documents using standards.",
  "key_deliverables": [
    "Rule-based gate validation",
    "Document generation and validation",
    "Approval decision report output"
  ],
  "known_risks": [
    "Delayed inputs from teams",
    "Scope changes causing budget variance",
    "Integration dependencies"
  ]
}'''

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PMO Agent", layout="wide",
                   initial_sidebar_state="collapsed", page_icon="⬡")

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in [("last_result", None), ("download_cache", {}), ("last_run_id", None),
             ("show_info", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── HELPERS ───────────────────────────────────────────────────────────────────
def _get(d, field, default=None):
    if isinstance(d, dict):
        return d.get(field, default)
    return getattr(d, field, default)

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if name.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            return "\n".join(pg.extract_text() for pg in reader.pages if pg.extract_text())
        except Exception:
            return data.decode("utf-8", errors="ignore")
    if name.endswith(".json"):
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {name}")

# ── STYLES ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
  .stApp { background: #06080f; color: #e2e8f0; }
  .block-container { max-width: 1100px; padding: 2rem 2rem 5rem 2rem !important; }

  /* Hide sidebar toggle + default header/footer */
  #MainMenu, footer, header { visibility: hidden !important; }
  [data-testid="collapsedControl"] { display: none !important; }
  [data-testid="stSidebar"] { display: none !important; }

  /* Inputs */
  .stTextInput input, .stNumberInput input {
    background: #0d1220 !important; border: 1px solid rgba(99,179,237,0.15) !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
  }
  .stTextInput input:focus, .stNumberInput input:focus {
    border-color: rgba(99,179,237,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.08) !important;
  }
  .stTextArea textarea {
    background: #0a0e1a !important; border: 1px solid rgba(99,179,237,0.15) !important;
    color: #c8d6e5 !important; border-radius: 10px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 12.5px !important;
    line-height: 1.7 !important; transition: border-color 0.2s, box-shadow 0.2s !important;
  }
  .stTextArea textarea:focus {
    border-color: rgba(99,179,237,0.45) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.07) !important;
  }

  /* Primary button */
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a56db 0%, #0ea5e9 100%) !important;
    border: none !important; color: #fff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 700 !important;
    font-size: 14px !important; letter-spacing: 0.04em !important;
    border-radius: 10px !important; padding: 0.65rem 1.6rem !important;
    width: 100% !important; box-shadow: 0 4px 28px rgba(14,165,233,0.3) !important;
    transition: all 0.25s ease !important;
  }
  .stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 36px rgba(14,165,233,0.5) !important;
    transform: translateY(-2px) !important;
  }

  /* Secondary button */
  .stButton > button:not([kind="primary"]) {
    background: transparent !important; border: 1px solid rgba(99,179,237,0.2) !important;
    color: #63b3ed !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important; border-radius: 8px !important;
    transition: all 0.2s !important;
  }
  .stButton > button:not([kind="primary"]):hover {
    background: rgba(99,179,237,0.07) !important;
    border-color: rgba(99,179,237,0.45) !important;
  }

  /* Download buttons */
  .stDownloadButton > button {
    background: transparent !important; border: 1px solid rgba(16,185,129,0.3) !important;
    color: #6ee7b7 !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; border-radius: 7px !important; width: 100% !important;
    padding: 6px 12px !important; transition: all 0.2s !important;
  }
  .stDownloadButton > button:hover {
    background: rgba(16,185,129,0.08) !important;
    border-color: rgba(16,185,129,0.6) !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: #0d1220 !important; border: 1px solid rgba(99,179,237,0.1) !important;
    border-radius: 8px !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12.5px !important; color: #90cdf4 !important;
  }
  .streamlit-expanderContent {
    background: #080c14 !important; border: 1px solid rgba(99,179,237,0.06) !important;
    border-top: none !important; border-radius: 0 0 8px 8px !important;
  }

  /* Alerts */
  .stAlert { border-radius: 8px !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; }
  div[data-testid="stSuccessMessage"] { background: rgba(16,185,129,0.08) !important; border: 1px solid rgba(16,185,129,0.25) !important; color: #6ee7b7 !important; }
  div[data-testid="stErrorMessage"]   { background: rgba(239,68,68,0.08) !important;   border: 1px solid rgba(239,68,68,0.25) !important; }
  div[data-testid="stInfoMessage"]    { background: rgba(99,179,237,0.06) !important;   border: 1px solid rgba(99,179,237,0.2) !important; }

  /* Dataframe */
  .stDataFrame { border: 1px solid rgba(99,179,237,0.1) !important; border-radius: 10px !important; overflow: hidden !important; }
  [data-testid="stDataFrame"] th { background: #0d1220 !important; color: #63b3ed !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
  [data-testid="stDataFrame"] td { background: #0a0f1a !important; color: #cbd5e0 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; }

  hr { border: none !important; border-top: 1px solid rgba(99,179,237,0.07) !important; margin: 1.5rem 0 !important; }

  /* Animations */
  @keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
  }
  .fade-up   { animation: fadeSlideUp 0.5s ease forwards; }
  .fade-up-2 { animation: fadeSlideUp 0.5s 0.1s ease both; }
  .fade-up-3 { animation: fadeSlideUp 0.5s 0.2s ease both; }

  /* File uploader */
  [data-testid="stFileUploaderDropzone"] {
    background: #0a0e1a !important; border: 1px dashed rgba(99,179,237,0.25) !important;
    border-radius: 10px !important; transition: border-color 0.2s !important;
  }
  [data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(99,179,237,0.5) !important;
  }

  /* Radio */
  [data-testid="stRadio"] label { font-size: 13px !important; color: #94a3b8 !important; }

  /* Info modal overlay */
  .info-modal {
    position: fixed; inset: 0; z-index: 9999;
    background: rgba(0,0,0,0.75); backdrop-filter: blur(4px);
    display: flex; align-items: center; justify-content: center;
    animation: fadeSlideUp 0.25s ease;
  }
  .info-modal-box {
    background: #0d1220; border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px; padding: 32px 36px; max-width: 820px; width: 90%;
    max-height: 80vh; overflow-y: auto;
    box-shadow: 0 24px 80px rgba(0,0,0,0.6);
  }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
col_logo, col_title, col_info, col_time = st.columns([0.06, 0.6, 0.12, 0.22])

with col_logo:
    st.markdown("""
    <div style="width:42px;height:42px;background:linear-gradient(135deg,#1a56db,#0ea5e9);
                border-radius:10px;display:flex;align-items:center;justify-content:center;
                font-size:20px;margin-top:4px;">⬡</div>""", unsafe_allow_html=True)

with col_title:
    st.markdown("""
    <div class="fade-up">
      <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:24px;font-weight:800;
                  color:#f0f6ff;letter-spacing:-0.02em;line-height:1.1;">PMO Agent</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#7aa8cc;
                  margin-top:2px;letter-spacing:0.04em;">Governance Document Pipeline</div>
    </div>""", unsafe_allow_html=True)

with col_info:
    if st.button("ℹ  Guide", key="btn_info"):
        st.session_state.show_info = not st.session_state.show_info

with col_time:
    st.markdown(f"""
    <div style="text-align:right;padding-top:4px;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#6a90b8;
                  text-transform:uppercase;letter-spacing:0.1em;color:#8ab4d4;">Session</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#7ec8e3;margin-top:2px;">
        {datetime.now().strftime('%Y-%m-%d  %H:%M')}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ── INFO PANEL (shown inline when toggled) ────────────────────────────────────
if st.session_state.show_info:
    gate_colors = {"Before Start": "#6ee7b7", "Start Gate": "#fbbf24", "End Gate": "#a78bfa"}
    cards_html = ""
    for dtype, info in DOC_INFO.items():
        gc = gate_colors.get(info["gate"], "#63b3ed")
        secs = "".join(f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#7aabdf;padding:1px 0;">· {s}</div>' for s in info["sections"])
        cards_html += f"""
        <div style="background:#080c14;border:1px solid rgba(99,179,237,0.1);border-radius:10px;
                    padding:16px 18px;flex:1;min-width:260px;max-width:320px;">
          <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:700;
                      color:#e2e8f0;margin-bottom:6px;">{info['title']}</div>
          <span style="background:rgba(99,179,237,0.08);border:1px solid rgba(99,179,237,0.15);
                       border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:9px;
                       color:{gc};padding:2px 8px;display:inline-block;margin-bottom:10px;">{info['gate']}</span>
          <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:12px;color:#a0b4c8;
                      margin-bottom:10px;line-height:1.55;">{info['desc']}</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#6a90b8;
                      text-transform:uppercase;letter-spacing:0.06em;margin-bottom:5px;">Required Sections</div>
          {secs}
        </div>"""

    st.markdown(f"""
    <div class="fade-up" style="background:#0d1220;border:1px solid rgba(99,179,237,0.15);
                border-radius:14px;padding:28px 32px;margin-bottom:24px;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:18px;font-weight:700;color:#f0f6ff;">
          Required Governance Documents
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#7aa8cc;">
          Upload what you have — the rest will be generated
        </div>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:14px;">{cards_html}</div>
      <div style="margin-top:20px;padding:14px 18px;background:rgba(99,179,237,0.05);
                  border:1px solid rgba(99,179,237,0.1);border-radius:8px;">
        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:13px;color:#90cdf4;font-weight:600;margin-bottom:6px;">
          Project Details Required
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;">
          {"".join(f'<span style="background:#0a0e1a;border:1px solid rgba(99,179,237,0.12);border-radius:5px;font-family:IBM Plex Mono,monospace;font-size:10px;color:#63b3ed;padding:3px 9px;">{f}</span>' for f in ["project_id","project_name","project_type","sponsor","estimated_budget","actual_budget_consumed","total_time_taken_days","timeline_summary","scope_summary","key_deliverables","known_risks"])}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── SECTION 1: PROJECT DETAILS ────────────────────────────────────────────────
st.markdown("""
<div class="fade-up" style="margin-bottom:8px;">
  <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:20px;font-weight:700;
              color:#f0f6ff;letter-spacing:-0.01em;">Project Details</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#7aa8cc;margin-top:4px;">
    Provide project information by uploading a file or entering the JSON directly.
  </div>
</div>""", unsafe_allow_html=True)

proj_col1, proj_col2 = st.columns([1, 1], gap="large")

with proj_col1:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#7aa8cc;
                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">
      Upload Project File
    </div>""", unsafe_allow_html=True)
    proj_file = st.file_uploader(
        "Upload project details (.json, .pdf, .docx)",
        type=["json", "pdf", "docx", "txt"],
        key="proj_upload",
        label_visibility="collapsed",
        help="Upload a JSON file with project details, or a PDF/Word document — the system will extract the context automatically.",
    )
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#6a90b8;
                margin-top:8px;line-height:1.6;">
      Supported formats: .json · .pdf · .docx · .txt<br/>
      JSON will be parsed directly. PDF/DOCX text will be extracted by the AI.
    </div>""", unsafe_allow_html=True)

with proj_col2:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#7aa8cc;
                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">
      Or Enter JSON Directly
    </div>""", unsafe_allow_html=True)
    proj_json_input = st.text_area(
        "Project JSON",
        value=SAMPLE_PROJECT_JSON,
        height=220,
        label_visibility="collapsed",
        help="Paste your project details as JSON. All fields are optional — missing ones will be extracted from uploaded documents.",
        key="proj_json",
    )

# Resolve project from input
manual_project = None
raw_upload_text = ""

if proj_file is not None:
    try:
        file_text = extract_text(proj_file)
        if proj_file.name.endswith(".json"):
            try:
                data = json.loads(file_text)
                manual_project = Project(**{k: v for k, v in data.items() if hasattr(Project(), k)})
                st.success(f"✓  Loaded project: {manual_project.project_name}")
            except Exception:
                raw_upload_text = file_text
                st.info("File content will be parsed by the AI to extract project details.")
        else:
            raw_upload_text = file_text
            st.info("Document uploaded — AI will extract project context automatically.")
    except Exception as e:
        st.error(f"Could not read file: {e}")

if manual_project is None and proj_json_input.strip():
    try:
        data = json.loads(proj_json_input)
        manual_project = Project(**{k: v for k, v in data.items() if hasattr(Project(), k)})
    except Exception:
        pass  # Will fall back to extraction

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# ── SECTION 2: GOVERNANCE DOCUMENTS ──────────────────────────────────────────
st.markdown("""
<div class="fade-up-2" style="margin-bottom:8px;">
  <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:20px;font-weight:700;
              color:#f0f6ff;letter-spacing:-0.01em;">Governance Documents</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#7aa8cc;margin-top:4px;">
    Upload any documents already available. Missing documents will be generated automatically.
  </div>
</div>""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload governance documents",
    type=["txt", "md", "docx", "pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
    help="Upload Proof of Value, Risk Assessment, or any other governance documents you already have.",
)

uploaded_mapping: dict = {}

if uploaded_files:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#7aa8cc;
                text-transform:uppercase;letter-spacing:0.08em;margin:10px 0 8px 0;">
      Map each file to its document type
    </div>""", unsafe_allow_html=True)
    for uf in uploaded_files:
        c1, c2 = st.columns([2, 2])
        with c1:
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:12px;"
                f"color:#90cdf4;padding:9px 0;'>📄 {uf.name}</div>",
                unsafe_allow_html=True,
            )
        with c2:
            dtype = st.selectbox(
                "Type", DOC_TYPES,
                format_func=lambda x: DOC_INFO[x]["title"],
                key=f"dtype_{uf.name}",
                label_visibility="collapsed",
            )
        try:
            txt = extract_text(uf)
            uploaded_mapping[dtype] = (uploaded_mapping.get(dtype, "") + "\n\n" + txt).strip()
        except Exception as e:
            st.error(f"{uf.name}: {e}")

    covered = set(uploaded_mapping.keys())
    missing = [DOC_INFO[d]["title"] for d in DOC_TYPES if d not in covered]
    if missing:
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#fbbf24;"
            f"margin-top:6px;'>⚡ Will be generated: {', '.join(missing)}</div>",
            unsafe_allow_html=True,
        )
else:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#6a90b8;
                margin-top:6px;">
      No files uploaded — all 6 governance documents will be generated automatically.
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# ── RUN BUTTON ────────────────────────────────────────────────────────────────
st.markdown("<div class='fade-up-3'>", unsafe_allow_html=True)
run_now = st.button("▶  Run PMO Agent", type="primary")
st.markdown("</div>", unsafe_allow_html=True)

if run_now:
    try:
        standards = load_standards(STANDARDS_PATH)
    except Exception as e:
        st.error(f"Failed to load standards: {e}")
        st.stop()

    project = manual_project if manual_project else Project()

    state = PMOState(project=project, standards=standards, provider=PROVIDER, model=MODEL)
    state.audit["api_key"]          = ""   # resolved from hardcoded default in llm_providers.py
    state.audit["uploaded_docs"]    = list(uploaded_mapping.keys())
    state.audit["raw_upload_text"]  = raw_upload_text + "\n\n" + "\n\n".join(uploaded_mapping.values())
    state.audit["uploaded_mapping"] = uploaded_mapping

    graph = build_graph()

    import llm_providers as _llmp
    _llmp.clear_status_log()

    status_line = st.empty()

    def _update_status(doc="", provider="", model="", note=""):
        if not doc:
            msg = "⏳  Starting PMO Agent…"
        else:
            msg = f"⏳  Generating <b>{doc}</b> — trying <b>{provider}</b> ({model})"
            if note:
                msg += f" · <span style='color:#fbbf24;'>{note[:60]}</span>"
        status_line.markdown(
            f'''<div style="background:#0a0e1a;border:1px solid rgba(99,179,237,0.15);
                        border-radius:8px;padding:10px 16px;font-family:IBM Plex Mono,monospace;
                        font-size:12px;color:#c4d4e4;">{msg}</div>''',
            unsafe_allow_html=True,
        )

    def _done_status(entries):
        if not entries:
            status_line.empty()
            return
        last = entries[-1]
        ok_count   = sum(1 for e in entries if e["status"] == "ok")
        fall_count = sum(1 for e in entries if e["status"] == "fallback")
        providers_used = list(dict.fromkeys(e["provider"] for e in entries if e["status"] == "ok"))
        prov_str = ", ".join(providers_used) if providers_used else "Local Template"
        icon = "✓" if ok_count > 0 else "⚠"
        color = "#6ee7b7" if fall_count == 0 else "#fbbf24"
        status_line.markdown(
            f'''<div style="background:#0a0e1a;border:1px solid rgba(99,179,237,0.15);
                        border-radius:8px;padding:10px 16px;font-family:IBM Plex Mono,monospace;
                        font-size:12px;color:#c4d4e4;">
              <span style="color:{color};">{icon}</span>
              &nbsp;Generation complete — used: <b style="color:{color};">{prov_str}</b>
              &nbsp;·&nbsp;{ok_count} docs generated
              {f'&nbsp;·&nbsp;<span style="color:#fbbf24;">{fall_count} fallback(s)</span>' if fall_count else ""}
            </div>''',
            unsafe_allow_html=True,
        )

    # Monkey-patch _log so we get live updates during graph execution
    _orig_log = _llmp._log
    def _patched_log(doc_type, provider, model, status, note=""):
        _orig_log(doc_type, provider, model, status, note)
        if status == "fallback":
            _update_status(doc_type, provider, model, note)
        elif status == "ok":
            _update_status(doc_type, provider, model)
    _llmp._log = _patched_log

    _update_status()

    try:
        raw_result = graph.invoke(state)
        _done_status(_llmp.get_status_log())
    except Exception as e:
        _done_status(_llmp.get_status_log())
        err_msg = str(e)
        if "rate_limit" in err_msg.lower() or "429" in err_msg or "fallback" in err_msg.lower():
            st.warning("⚠  API limit reached — documents generated using local template.")
            raw_result = state
        else:
            st.error(f"Agent run failed: {e}")
            st.stop()
    finally:
        _llmp._log = _orig_log  # restore original

    try:
        result = PMOState.model_validate(raw_result)
    except Exception:
        try:
            result = PMOState(**(dict(raw_result) if not isinstance(raw_result, dict) else raw_result))
        except Exception as e:
            st.error(f"Could not parse result: {e}")
            st.stop()

    st.session_state.last_result = result
    st.session_state.last_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Cache all output bytes immediately
    cache: dict = {}
    from doc_templates import create_decision_report_docx, create_official_docx

    org    = standards["org"]["name"]
    header = standards["org"]["doc_header"]
    footer = standards["org"]["doc_footer"]

    cache[f"{result.project.project_id}_PMO_Decision_Report.docx"] = create_decision_report_docx(
        org_name=org, header=header, footer=footer,
        project_id=result.project.project_id, project_name=result.project.project_name,
        decision=result.decision or "UNKNOWN", summary=result.summary or "",
        gates=result.gates, docs=result.docs,
    )

    required_keys = set(
        result.required_docs.get("BEFORE_START", []) +
        result.required_docs.get("START", []) +
        result.required_docs.get("END", [])
    )

    for k, d in result.docs.items():
        content = _get(d, "content_markdown", "") or ""
        title   = _get(d, "title", k) or k
        if content.strip():
            cache[f"{result.project.project_id}_{k}.docx"] = create_official_docx(
                org_name=org, header=header, footer=footer, title=title, md_content=content,
            )
            cache[f"{result.project.project_id}_{k}.md"] = content.encode("utf-8")

    st.session_state.download_cache = cache

# ── RESULTS ───────────────────────────────────────────────────────────────────
result = st.session_state.last_result

if result is not None:
    decision   = str(result.decision or "PENDING").strip().upper()
    is_invalid = any(w in decision for w in ("INVALID", "REJECT", "FAIL", "DENY"))
    is_approve = any(w in decision for w in ("APPROV", "PASS"))

    docs_total = len(result.docs)
    docs_ok    = sum(1 for d in result.docs.values() if _get(d, "status") == "SUFFICIENT")
    pass_pct   = int((docs_ok / docs_total) * 100) if docs_total else 0

    est       = getattr(result.project, "estimated_budget", None)
    actual    = getattr(result.project, "actual_budget_consumed", None)
    budget_ok = est is not None and actual is not None and actual <= est

    dec_color = "#6ee7b7" if is_approve else ("#f87171" if is_invalid else "#fbbf24")

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:11px;font-weight:600;
                color:#6a90b8;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:14px;">
      Results
    </div>""", unsafe_allow_html=True)

    # ── KPI CARDS ─────────────────────────────────────────────────────────────
    def kpi(label, value, sub, accent):
        return f"""
        <div style="background:#0d1220;border:1px solid rgba(99,179,237,0.1);border-radius:12px;
                    padding:20px 22px;position:relative;overflow:hidden;
                    animation:fadeSlideUp 0.4s ease;">
          <div style="position:absolute;top:0;left:0;right:0;height:2px;
                      background:linear-gradient(90deg,{accent},transparent);"></div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#7aa8cc;
                      text-transform:uppercase;letter-spacing:0.12em;margin-bottom:12px;">{label}</div>
          <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:28px;font-weight:800;
                      color:{accent};line-height:1;">{value}</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#6a90b8;
                      margin-top:8px;">{sub}</div>
        </div>"""

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1: st.markdown(kpi("Decision", decision, "Final PMO verdict", dec_color), unsafe_allow_html=True)
    with k2: st.markdown(kpi("Documents", f"{docs_ok}/{docs_total}", "Passed validation", "#63b3ed"), unsafe_allow_html=True)
    with k3:
        pa = "#f87171" if is_invalid else ("#6ee7b7" if pass_pct == 100 else "#a78bfa")
        st.markdown(kpi("Pass Rate", f"{pass_pct}%", "Sufficiency score", pa), unsafe_allow_html=True)
    with k4:
        bd = f"{actual:,.0f}" if actual is not None else "—"
        bs = f"Est: {est:,.0f}" if est is not None else "No estimate"
        st.markdown(kpi("Actual Spend", bd, bs, "#6ee7b7" if budget_ok else "#fbbf24"), unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── DOCS + DOWNLOADS ──────────────────────────────────────────────────────
    col_docs, col_dl = st.columns([3, 1], gap="large")
    cache = st.session_state.get("download_cache", {})

    with col_docs:
        st.markdown("""
        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:15px;font-weight:700;
                    color:#90cdf4;margin-bottom:14px;padding-bottom:8px;
                    border-bottom:1px solid rgba(99,179,237,0.08);">Document Review</div>""",
                    unsafe_allow_html=True)

        req_keys = set(
            result.required_docs.get("BEFORE_START", []) +
            result.required_docs.get("START", []) +
            result.required_docs.get("END", [])
        )

        try:
            import pandas as pd
            rows = []
            for key, d in result.docs.items():
                status  = _get(d, "status") or "NOT_AVAILABLE"
                preview = (_get(d, "content_markdown") or "").strip().replace("\n", " ")
                preview = (preview[:MAX_PREVIEW] + "…") if len(preview) > MAX_PREVIEW else preview
                rows.append({
                    "Document":  DOC_INFO.get(key, {}).get("title", key),
                    "Status":    status,
                    "Gate":      "Required" if key in req_keys else "Optional",
                    "Preview":   preview,
                })
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        except Exception:
            pass

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        for key, d in result.docs.items():
            status  = _get(d, "status") or "NOT_AVAILABLE"
            content = _get(d, "content_markdown") or ""
            reasons = _get(d, "reasons") or []
            title   = DOC_INFO.get(key, {}).get("title", key)

            if status == "SUFFICIENT":
                badge = "<span style='background:rgba(110,231,183,0.1);color:#6ee7b7;border:1px solid rgba(110,231,183,0.2);padding:2px 8px;border-radius:4px;font-size:10px;font-family:IBM Plex Mono,monospace;'>SUFFICIENT</span>"
            else:
                badge = f"<span style='background:rgba(248,113,113,0.1);color:#f87171;border:1px solid rgba(248,113,113,0.2);padding:2px 8px;border-radius:4px;font-size:10px;font-family:IBM Plex Mono,monospace;'>{status}</span>"
            if key in req_keys:
                badge += " <span style='background:rgba(99,179,237,0.1);color:#63b3ed;border:1px solid rgba(99,179,237,0.15);padding:2px 8px;border-radius:4px;font-size:10px;font-family:IBM Plex Mono,monospace;'>REQUIRED</span>"

            with st.expander(title, expanded=False):
                st.markdown(f"<div style='margin-bottom:8px;'>{badge}</div>", unsafe_allow_html=True)
                for rr in reasons:
                    st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#a0b4c8;padding:3px 0;'>→ {rr}</div>", unsafe_allow_html=True)
                if content:
                    preview_text = content[:900] + "…" if len(content) > 900 else content
                    st.markdown(
                        f"<div style='background:#0a0f1a;border:1px solid rgba(99,179,237,0.08);border-radius:8px;"
                        f"padding:14px 16px;margin-top:8px;font-family:IBM Plex Mono,monospace;"
                        f"font-size:12px;color:#c4d4e4;line-height:1.75;white-space:pre-wrap;'>"
                        f"{preview_text}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No content available.")

    with col_dl:
        st.markdown("""
        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:15px;font-weight:700;
                    color:#90cdf4;margin-bottom:14px;padding-bottom:8px;
                    border-bottom:1px solid rgba(99,179,237,0.08);">Downloads</div>""",
                    unsafe_allow_html=True)

        report_fname = f"{result.project.project_id}_PMO_Decision_Report.docx"
        if report_fname in cache:
            st.download_button("⬇  Decision Report", data=cache[report_fname],
                               file_name=report_fname, key="dl_report")

        if cache:
            zb = io.BytesIO()
            with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as z:
                for fn, b in cache.items():
                    z.writestr(fn, b)
            zb.seek(0)
            st.download_button("⬇  All Outputs (ZIP)", data=zb,
                               file_name=f"PMO_{st.session_state.last_run_id}.zip",
                               mime="application/zip", key="dl_zip")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#6a90b8;
                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">
          Individual Documents
        </div>""", unsafe_allow_html=True)

        for k, d in result.docs.items():
            dst   = _get(d, "status") or "—"
            dot   = "🟢" if dst == "SUFFICIENT" else "🔴"
            label = DOC_INFO.get(k, {}).get("title", k)
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:10px;"
                f"color:#7aabdf;margin-bottom:4px;'>{dot} {label}</div>",
                unsafe_allow_html=True,
            )
            docx_f = f"{result.project.project_id}_{k}.docx"
            md_f   = f"{result.project.project_id}_{k}.md"
            c1, c2 = st.columns(2)
            if docx_f in cache:
                with c1:
                    st.download_button("DOCX", data=cache[docx_f], file_name=docx_f,
                                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                       key=f"dl_{docx_f}")
            if md_f in cache:
                with c2:
                    st.download_button("MD", data=cache[md_f], file_name=md_f,
                                       mime="text/markdown", key=f"dl_{md_f}")

    # ── AUDIT ─────────────────────────────────────────────────────────────────
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:15px;font-weight:700;
                color:#90cdf4;margin-bottom:12px;">Audit Trail</div>""", unsafe_allow_html=True)

    with st.expander("View full audit log", expanded=False):
        st.json(result.audit if isinstance(result.audit, dict) else dict(result.audit))

    # ── COMPLETE BANNER ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-top:28px;padding:16px 22px;background:rgba(16,185,129,0.06);
                border:1px solid rgba(16,185,129,0.2);border-radius:10px;
                display:flex;align-items:center;gap:14px;animation:fadeSlideUp 0.5s ease;">
      <div style="font-size:20px;">✓</div>
      <div>
        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:700;
                    color:#6ee7b7;">Run complete — {result.project.project_id}</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#5ab890;margin-top:3px;">
          Decision: {decision}  ·  {docs_ok}/{docs_total} documents passed  ·  Download the reports above.
        </div>
      </div>
    </div>""", unsafe_allow_html=True)