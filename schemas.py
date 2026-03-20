from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field

DocStatus = Literal["NOT_AVAILABLE", "NOT_SUFFICIENT", "SUFFICIENT", "SUFFICIENT_WITH_FLAGS"]

class Project(BaseModel):
    project_id: str = "TBD"
    project_name: str = "TBD"
    project_type: str = "Default"
    sponsor: Optional[str] = None
    estimated_budget: Optional[float] = None
    actual_budget_consumed: Optional[float] = None
    total_time_taken_days: Optional[int] = None
    timeline_summary: Optional[str] = None
    scope_summary: Optional[str] = None
    key_deliverables: List[str] = Field(default_factory=list)
    known_risks: List[str] = Field(default_factory=list)

class DocumentArtifact(BaseModel):
    doc_type: str
    title: str
    content_markdown: str = ""
    status: DocStatus = "NOT_AVAILABLE"
    reasons: List[str] = Field(default_factory=list)
    file_path: Optional[str] = None  # exported docx path

class GateResult(BaseModel):
    gate: str
    passed: bool
    findings: List[str] = Field(default_factory=list)

class PMOState(BaseModel):
    project: Project
    standards: Dict
    provider: str
    model: str

    docs: Dict[str, DocumentArtifact] = Field(default_factory=dict)
    required_docs: Dict[str, List[str]] = Field(default_factory=dict)

    gates: List[GateResult] = Field(default_factory=list)
    decision: Optional[str] = None
    summary: Optional[str] = None
    audit: Dict[str, Any] = Field(default_factory=dict)