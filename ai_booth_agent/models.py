from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class BoothRequest(BaseModel):
    company_name: str
    booth_size_sqm: float
    budget_inr: float
    event_name: str
    event_date: str
    industry: str
    location: Optional[str] = None
    past_events: Optional[list[str]] = None

class AgentResponse(BaseModel):
    summary: str
    approval_recommendation: str
    competitor_analysis: Optional[str]
    risk_flags: list[str]
    analytics: Optional[dict] = None

class BoothElement(BaseModel):
    id: str
    type: str
    position: List[float]
    size: List[float]
    color: str
    label: str

class BoothLighting(BaseModel):
    ambient: float
    spotlights: List[Dict[str, Any]]

class BoothDimensions(BaseModel):
    width: float
    depth: float
    height: float

class BoothMaterials(BaseModel):
    floor: str
    walls: str
    ceiling: str

class BoothColors(BaseModel):
    primary: str
    secondary: str
    accent: str

class BoothDesignModel(BaseModel):
    dimensions: BoothDimensions
    materials: BoothMaterials
    colors: BoothColors
    elements: List[BoothElement]
    lighting: BoothLighting

class BoothDesignRequest(BaseModel):
    session_id: str
    design_name: str
    design_data: BoothDesignModel