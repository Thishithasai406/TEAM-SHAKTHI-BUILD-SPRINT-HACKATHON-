import datetime
from typing import Dict, Any, Optional

def normalize_deal(raw_data: Dict[str, Any], source_type: str = "custom") -> Dict[str, Any]:
    """
    Normalizes any raw input dict into the canonical DealIQ deal dictionary schema.
    Safely preserves None/null for unavailable CRM signals.
    """
    now_iso = datetime.datetime.utcnow().isoformat()
    
    deal_id = str(raw_data.get("deal_id") or raw_data.get("id") or raw_data.get("Id") or "").strip()
    
    raw_account = raw_data.get("company") or raw_data.get("company_name") or raw_data.get("Account") or "Unknown Company"
    if isinstance(raw_account, dict):
        company = str(raw_account.get("Name") or "Unknown Company").strip()
    else:
        company = str(raw_account).strip()
    
    try:
        deal_size = float(raw_data.get("deal_size") or raw_data.get("Amount") or raw_data.get("amount") or 0.0)
    except (ValueError, TypeError):
        deal_size = 0.0

    stage = str(raw_data.get("stage") or raw_data.get("StageName") or "Proposal").strip()
    
    try:
        days_in_stage = int(raw_data.get("days_in_stage") if raw_data.get("days_in_stage") is not None else 14)
    except (ValueError, TypeError):
        days_in_stage = 14

    # Optional CRM Signals (Safe None handling)
    def parse_opt_float(val: Any) -> Optional[float]:
        if val is None or val == "" or str(val).lower() in ["null", "none", "n/a"]:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def parse_opt_int(val: Any) -> Optional[int]:
        if val is None or val == "" or str(val).lower() in ["null", "none", "n/a"]:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    response_latency_hrs = parse_opt_float(raw_data.get("response_latency_hrs"))
    
    raw_sentiment = raw_data.get("sentiment_trend")
    if raw_sentiment and str(raw_sentiment).lower().strip() in ["improving", "stable", "declining"]:
        sentiment_trend = str(raw_sentiment).lower().strip()
    else:
        sentiment_trend = None

    stakeholder_count = parse_opt_int(raw_data.get("stakeholder_count"))
    competitor_mentions = parse_opt_int(raw_data.get("competitor_mentions"))
    scope_change_flags = parse_opt_int(raw_data.get("scope_change_flags"))

    outcome = str(raw_data.get("outcome", "in_progress")).lower().strip()
    source_record_id = str(raw_data.get("source_record_id") or deal_id)

    return {
        "deal_id": deal_id,
        "company": company,
        "company_name": company,  # legacy compatibility
        "deal_size": deal_size,
        "stage": stage,
        "days_in_stage": days_in_stage,
        "response_latency_hrs": response_latency_hrs,
        "sentiment_trend": sentiment_trend,
        "stakeholder_count": stakeholder_count,
        "competitor_mentions": competitor_mentions,
        "scope_change_flags": scope_change_flags,
        "outcome": outcome,
        "created_at": str(raw_data.get("created_at") or now_iso),
        "updated_at": str(raw_data.get("updated_at") or now_iso),
        "source": str(raw_data.get("source") or source_type),
        "source_record_id": source_record_id,
        "owner": raw_data.get("owner"),
        "close_date": raw_data.get("close_date"),
        "probability": parse_opt_float(raw_data.get("probability")),
        "currency": str(raw_data.get("currency") or "USD"),
        "description": raw_data.get("description"),
        "last_activity_at": raw_data.get("last_activity_at"),
        "contact_count": parse_opt_int(raw_data.get("contact_count")),
        "last_synced_at": str(raw_data.get("last_synced_at") or now_iso),
        "source_url": raw_data.get("source_url")
    }
