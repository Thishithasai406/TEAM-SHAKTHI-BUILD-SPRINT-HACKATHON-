import hmac
import hashlib
import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, Header, Depends, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from data_sources.provider_factory import get_active_provider, get_active_data_source_type, PROVIDERS
from data_sources.normalizer import normalize_deal
from data_sources.file_import import FileImportDataProvider

router = APIRouter(prefix="/api", tags=["Data Sources & Webhooks"])

class WebhookDealPayload(BaseModel):
    deal_id: str
    company: Optional[str] = None
    company_name: Optional[str] = None
    deal_size: Optional[float] = 0.0
    amount: Optional[float] = None
    stage: Optional[str] = "Proposal"
    days_in_stage: Optional[int] = 14
    response_latency_hrs: Optional[float] = None
    sentiment_trend: Optional[str] = None
    stakeholder_count: Optional[int] = None
    competitor_mentions: Optional[int] = None
    scope_change_flags: Optional[int] = None
    outcome: Optional[str] = "in_progress"
    source: Optional[str] = "webhook"
    source_record_id: Optional[str] = None

@router.get("/data-source/status")
def get_data_source_status():
    """Returns status and connectivity information for the configured data provider."""
    provider = get_active_provider()
    source_type = get_active_data_source_type()
    check = provider.health_check()
    
    # Overview of all available providers
    all_statuses = {}
    for name, p_inst in PROVIDERS.items():
        all_statuses[name] = p_inst.health_check()
        
    return {
        "active_source": source_type,
        "active_provider_status": check,
        "all_providers": all_statuses,
        "last_synced_at": check.get("last_synced_at") or datetime.datetime.utcnow().isoformat()
    }

@router.post("/data-source/sync")
def sync_data_source():
    """Triggers an explicit re-sync of the active data provider."""
    provider = get_active_provider()
    source_type = get_active_data_source_type()
    deals = provider.get_deals()
    check = provider.health_check()
    
    return {
        "message": f"Successfully synced {len(deals)} deals from {source_type}",
        "active_source": source_type,
        "deals_synced": len(deals),
        "synced_at": datetime.datetime.utcnow().isoformat(),
        "status": check
    }

@router.post("/import/csv")
@router.post("/import/excel")
async def import_pipeline_file(file: UploadFile = File(...)):
    """Uploads and parses a CSV or Excel pipeline file, normalizing rows into canonical schema."""
    filename = file.filename or "uploaded_pipeline.csv"
    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .csv, .xlsx, and .xls files are supported.")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File size exceeds maximum 10MB limit.")

    file_provider: FileImportDataProvider = PROVIDERS["file"]
    valid_deals, summary = file_provider.process_file_content(filename, content)

    return {
        "message": f"File import completed: {summary['valid_rows']} valid rows imported.",
        "filename": filename,
        "summary": summary,
        "deals": valid_deals[:10]  # Return sample of first 10 imported deals
    }

@router.post("/webhooks/deals")
async def handle_deal_webhook(
    request: Request,
    payload: WebhookDealPayload,
    x_webhook_signature: Optional[str] = Header(None)
):
    """
    Idempotent Webhook endpoint for near-real-time CRM deal updates.
    Updates or inserts deal in live cache and recalculates ML risk & recommendations.
    """
    # Optional HMAC signature validation if WEBHOOK_SECRET is set
    webhook_secret = os.getenv("WEBHOOK_SECRET")
    if webhook_secret:
        if not x_webhook_signature:
            raise HTTPException(status_code=401, detail="Missing X-Webhook-Signature header")
        raw_body = await request.body()
        expected_sig = hmac.new(webhook_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_sig, x_webhook_signature):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    raw_dict = payload.dict()
    canonical = normalize_deal(raw_dict, source_type=payload.source or "webhook")

    from main import PROCESSED_LIVE_DEALS, LIVE_DEALS_CACHE, process_single_deal, LAST_REFRESHED_ISO
    import main

    # Idempotent update or insert
    deal_id = canonical["deal_id"]
    existing_idx = None
    for idx, d in enumerate(LIVE_DEALS_CACHE):
        if d.get("deal_id") == deal_id or d.get("source_record_id") == deal_id:
            existing_idx = idx
            break

    now_iso = datetime.datetime.utcnow().isoformat()
    canonical["last_synced_at"] = now_iso
    processed = process_single_deal(canonical)

    if existing_idx is not None:
        LIVE_DEALS_CACHE[existing_idx] = canonical
        PROCESSED_LIVE_DEALS[existing_idx] = processed
        action = "updated"
    else:
        LIVE_DEALS_CACHE.append(canonical)
        PROCESSED_LIVE_DEALS.append(processed)
        action = "created"

    main.LAST_REFRESHED_ISO = now_iso

    return {
        "status": "success",
        "action": action,
        "deal_id": deal_id,
        "processed_deal": processed,
        "timestamp": now_iso
    }
