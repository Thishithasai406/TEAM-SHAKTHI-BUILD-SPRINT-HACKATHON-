import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class CanonicalDeal(BaseModel):
    # Required Fields
    deal_id: str
    company: str
    deal_size: float
    stage: str
    days_in_stage: int
    
    # Optional / CRM-dependent fields (Can be None if unavailable)
    response_latency_hrs: Optional[float] = None
    sentiment_trend: Optional[str] = None  # "improving", "stable", "declining", or None
    stakeholder_count: Optional[int] = None
    competitor_mentions: Optional[int] = None
    scope_change_flags: Optional[int] = None
    
    outcome: str = "in_progress"  # "won", "stalled", "lost", "in_progress"
    created_at: str
    updated_at: str
    source: str  # "salesforce", "hubspot", "file", "synthetic"
    source_record_id: str
    
    # Optional metadata
    owner: Optional[str] = None
    close_date: Optional[str] = None
    probability: Optional[float] = None
    currency: Optional[str] = "USD"
    description: Optional[str] = None
    last_activity_at: Optional[str] = None
    contact_count: Optional[int] = None
    last_synced_at: Optional[str] = None
    source_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = self.dict()
        # Add legacy compatible company_name key
        d["company_name"] = self.company
        return d
