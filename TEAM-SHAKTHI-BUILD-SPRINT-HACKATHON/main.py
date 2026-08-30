import json
import random
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from feature_extractor import extract_engagement_features, EngagementFeatureVector
from deal_scorer import scorer
from recommendation_engine import generate_recovery_recommendation
from data_sources.provider_factory import get_active_provider, get_active_data_source_type
from api_router import router as data_sources_router

app = FastAPI(title="DealIQ API", description="Deal Intelligence and Health Scoring Engine")
app.include_router(data_sources_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

# Global state for in-memory live deal pipeline & cache
LIVE_DEALS_CACHE: List[Dict[str, Any]] = []
PROCESSED_LIVE_DEALS: List[Dict[str, Any]] = []
LAST_REFRESHED_ISO: str = datetime.now(timezone.utc).isoformat()


def load_json_data(file_name: str):
    file_path = DATA_DIR / file_name
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)
    return []


def format_currency(val: float) -> str:
    return f"${val:,.0f}"

STAGE_BENCHMARKS = {
    "Discovery": 14,
    "Qualification": 21,
    "Proposal": 30,
    "Negotiation": 21,
    "Closing": 14,
    "Evaluation": 30  # Fallback/alias if present
}

def calculate_deal_health(deal: dict) -> dict:
    """
    Calculates a transparent, deterministic, 0-100 Health Score based on observable deal features.
    Health Score breakdown:
    - Base score: 50 points
    - 6 Core Feature Categories (Max sum of contributions: +50 to -50):
      1. Stage Velocity (Benchmark-driven): +/- 10 pts
      2. Response Latency: +/- 10 pts
      3. Sentiment Trend: +/- 10 pts
      4. Stakeholder Coverage: +/- 10 pts
      5. Competitor Presence: +/- 5 pts
      6. Scope Stability: +/- 5 pts
    Total range before clamping: 0 to 100. Final score is clamped to [0.0, 100.0] and rounded to 1 decimal place.
    """
    stage = str(deal.get("stage", "Proposal")).strip()
    # Normalize stage casing for lookup
    stage_matched = "Proposal"
    for s_key in STAGE_BENCHMARKS:
        if s_key.lower() == stage.lower():
            stage_matched = s_key
            break

    benchmark_days = STAGE_BENCHMARKS.get(stage_matched, 30)
    days = int(deal.get("days_in_stage", 0))

    health_factors = []
    total_contribution = 0.0

    # 1. Stage Velocity (Max +10 / -10)
    if days <= benchmark_days:
        sv_score = 10.0
        impact = "positive"
        desc = f"Stage duration ({days}d) is within benchmark ({benchmark_days}d)"
    elif days <= benchmark_days * 1.5:
        sv_score = 4.0
        impact = "positive"
        desc = f"Stage duration ({days}d) slightly exceeds benchmark ({benchmark_days}d)"
    elif days <= benchmark_days * 2.0:
        sv_score = -4.0
        impact = "negative"
        desc = f"Stage duration ({days}d) moderately exceeds benchmark ({benchmark_days}d)"
    elif days <= benchmark_days * 3.0:
        sv_score = -8.0
        impact = "negative"
        desc = f"Stage duration ({days}d) significantly exceeds benchmark ({benchmark_days}d)"
    else:
        sv_score = -10.0
        impact = "negative"
        desc = f"Stage duration ({days}d) severely exceeds benchmark ({benchmark_days}d)"

    total_contribution += sv_score
    health_factors.append({
        "factor": "Stage Velocity",
        "impact": impact,
        "score_contribution": round(sv_score, 1),
        "value": f"{days} days",
        "benchmark": f"<= {benchmark_days} days ({stage_matched})"
    })

    # 2. Response Latency (Max +10 / -10)
    latency_raw = deal.get("response_latency_hrs")
    if latency_raw is not None:
        latency = float(latency_raw)
        if latency <= 12.0:
            rl_score = 10.0
            impact = "positive"
        elif latency <= 24.0:
            rl_score = 6.0
            impact = "positive"
        elif latency <= 48.0:
            rl_score = -4.0
            impact = "negative"
        elif latency <= 72.0:
            rl_score = -8.0
            impact = "negative"
        else:
            rl_score = -10.0
            impact = "negative"

        total_contribution += rl_score
        health_factors.append({
            "factor": "Response Latency",
            "impact": impact,
            "score_contribution": round(rl_score, 1),
            "value": f"{latency:.1f} hours",
            "benchmark": "<= 24.0 hours"
        })
    else:
        health_factors.append({
            "factor": "Response Latency",
            "impact": "neutral",
            "score_contribution": 0.0,
            "value": "Not available from CRM",
            "benchmark": "<= 24.0 hours"
        })

    # 3. Sentiment Trend (Max +10 / -10)
    sentiment_raw = deal.get("sentiment_trend")
    if sentiment_raw is not None:
        sentiment = str(sentiment_raw).lower().strip()
        if sentiment == "improving":
            st_score = 10.0
            impact = "positive"
        elif sentiment == "stable":
            st_score = 2.0
            impact = "positive"
        else:  # declining
            st_score = -10.0
            impact = "negative"

        total_contribution += st_score
        health_factors.append({
            "factor": "Sentiment Trend",
            "impact": impact,
            "score_contribution": round(st_score, 1),
            "value": sentiment.capitalize(),
            "benchmark": "Improving or Stable"
        })
    else:
        health_factors.append({
            "factor": "Sentiment Trend",
            "impact": "neutral",
            "score_contribution": 0.0,
            "value": "Not available from CRM",
            "benchmark": "Improving or Stable"
        })

    # 4. Stakeholder Coverage (Max +10 / -10)
    stakeholders_raw = deal.get("stakeholder_count")
    if stakeholders_raw is not None:
        stakeholders = int(stakeholders_raw)
        deal_size = float(deal.get("deal_size", 0.0))

        if stakeholders >= 5:
            sc_score = 10.0
            impact = "positive"
        elif stakeholders >= 3:
            sc_score = 6.0
            impact = "positive"
        elif stakeholders == 2:
            sc_score = 0.0
            impact = "neutral"
        else:  # 0 or 1 stakeholder
            sc_score = -10.0
            impact = "negative"

        if deal_size >= 250000 and stakeholders < 2:
            sc_score = min(sc_score, -10.0)

        total_contribution += sc_score
        health_factors.append({
            "factor": "Stakeholder Coverage",
            "impact": impact if sc_score != 0 else "neutral",
            "score_contribution": round(sc_score, 1),
            "value": f"{stakeholders} stakeholders",
            "benchmark": ">= 3 stakeholders"
        })
    else:
        health_factors.append({
            "factor": "Stakeholder Coverage",
            "impact": "neutral",
            "score_contribution": 0.0,
            "value": "Not available from CRM",
            "benchmark": ">= 3 stakeholders"
        })

    # 5. Competitor Presence (Max +5 / -5)
    comps_raw = deal.get("competitor_mentions")
    if comps_raw is not None:
        comps = int(comps_raw)
        if comps == 0:
            cp_score = 5.0
            impact = "positive"
        elif comps == 1:
            cp_score = 1.0
            impact = "positive"
        elif comps == 2:
            cp_score = -2.0
            impact = "negative"
        else:  # 3+
            cp_score = -5.0
            impact = "negative"

        total_contribution += cp_score
        health_factors.append({
            "factor": "Competitor Presence",
            "impact": impact,
            "score_contribution": round(cp_score, 1),
            "value": f"{comps} mentions",
            "benchmark": "0 competitor mentions"
        })
    else:
        health_factors.append({
            "factor": "Competitor Presence",
            "impact": "neutral",
            "score_contribution": 0.0,
            "value": "Not available from CRM",
            "benchmark": "0 competitor mentions"
        })

    # 6. Scope Stability (Max +5 / -5)
    scope_flags_raw = deal.get("scope_change_flags")
    if scope_flags_raw is not None:
        scope_flags = int(scope_flags_raw)
        if scope_flags == 0:
            ss_score = 5.0
            impact = "positive"
        elif scope_flags == 1:
            ss_score = 1.0
            impact = "positive"
        elif scope_flags == 2:
            ss_score = -2.0
            impact = "negative"
        else:  # 3+
            ss_score = -5.0
            impact = "negative"

        total_contribution += ss_score
        health_factors.append({
            "factor": "Scope Stability",
            "impact": impact,
            "score_contribution": round(ss_score, 1),
            "value": f"{scope_flags} changes",
            "benchmark": "0 scope change flags"
        })
    else:
        health_factors.append({
            "factor": "Scope Stability",
            "impact": "neutral",
            "score_contribution": 0.0,
            "value": "Not available from CRM",
            "benchmark": "0 scope change flags"
        })

    # Base score of 50.0 + total contributions
    raw_calculated_score = 50.0 + total_contribution
    final_score = round(max(0.0, min(100.0, raw_calculated_score)), 1)

    # Risk level classification strictly based on Health Score thresholds
    if final_score >= 70.0:
        risk_level = "Low"
    elif final_score >= 45.0:
        risk_level = "Medium"
    else:
        risk_level = "High"

    win_probability = round(final_score / 100.0, 2)

    # Map contributors for backward compatibility
    positive_contributors = [
        {"signal": f"{f['factor']}: {f['value']}", "points": f"+{f['score_contribution']:.1f}"}
        for f in health_factors if f["score_contribution"] > 0
    ]
    negative_contributors = [
        {"signal": f"{f['factor']}: {f['value']}", "points": f"{f['score_contribution']:.1f}"}
        for f in health_factors if f["score_contribution"] < 0
    ]

    return {
        "health_score": final_score,
        "raw_health_score": round(raw_calculated_score, 1),
        "risk_level": risk_level,
        "win_probability": win_probability,
        "health_factors": health_factors,
        "positive_contributors": positive_contributors,
        "negative_contributors": negative_contributors
    }


def process_single_deal(deal: Dict[str, Any]) -> Dict[str, Any]:
    """Computes health, risk score, and recovery recommendations for a single live deal."""
    health = calculate_deal_health(deal)
    risk_scoring = scorer.score_deal(deal, k=5)
    rec = generate_recovery_recommendation(
        live_deal=deal,
        risk_score=risk_scoring["risk_score"],
        nearest_analogs=risk_scoring["nearest_analogs"]
    )
    if "last_updated" not in deal:
        deal["last_updated"] = datetime.now(timezone.utc).isoformat()
    
    return {**deal, **health, **risk_scoring, "recovery_strategy": rec}


def initialize_live_pipeline():
    global LIVE_DEALS_CACHE, PROCESSED_LIVE_DEALS, LAST_REFRESHED_ISO
    provider = get_active_provider()
    raw_live = provider.get_deals()
    if not raw_live:
        raw_live = load_json_data("live_deals.json")

    now_iso = datetime.now(timezone.utc).isoformat()
    
    LIVE_DEALS_CACHE = []
    PROCESSED_LIVE_DEALS = []
    
    for d in raw_live:
        d_copy = dict(d)
        if "last_updated" not in d_copy:
            d_copy["last_updated"] = now_iso
        LIVE_DEALS_CACHE.append(d_copy)
        PROCESSED_LIVE_DEALS.append(process_single_deal(d_copy))
        
    LAST_REFRESHED_ISO = now_iso


# Initialize pipeline on startup
initialize_live_pipeline()


async def live_simulation_scheduler():
    """Background task running every 60 seconds to nudge 5-10 random live deals."""
    global LIVE_DEALS_CACHE, PROCESSED_LIVE_DEALS, LAST_REFRESHED_ISO
    while True:
        await asyncio.sleep(60)
        try:
            if not LIVE_DEALS_CACHE:
                continue

            num_to_nudge = random.randint(5, 10)
            target_indices = random.sample(range(len(LIVE_DEALS_CACHE)), min(num_to_nudge, len(LIVE_DEALS_CACHE)))
            now_iso = datetime.now(timezone.utc).isoformat()

            for idx in target_indices:
                deal = LIVE_DEALS_CACHE[idx]
                
                # Apply random perturbation
                action = random.choice(["latency", "sentiment", "days", "scope"])
                if action == "latency":
                    deal["response_latency_hrs"] = round(max(0.5, deal.get("response_latency_hrs", 12.0) + random.uniform(-10, 15)), 1)
                elif action == "sentiment":
                    deal["sentiment_trend"] = random.choice(["improving", "stable", "declining"])
                elif action == "days":
                    deal["days_in_stage"] = deal.get("days_in_stage", 10) + random.randint(1, 5)
                elif action == "scope":
                    deal["scope_change_flags"] = max(0, deal.get("scope_change_flags", 0) + random.choice([-1, 1]))

                deal["last_updated"] = now_iso
                
                # Re-process ONLY the affected deal
                PROCESSED_LIVE_DEALS[idx] = process_single_deal(deal)

            LAST_REFRESHED_ISO = now_iso
        except Exception as e:
            print(f"[Simulation Scheduler Error]: {e}")


@app.on_event("startup")
async def start_background_simulation():
    asyncio.create_task(live_simulation_scheduler())


@app.get("/api/deals/historical")
def get_historical_deals(
    stage: Optional[str] = None,
    outcome: Optional[str] = None,
    min_size: Optional[float] = None
):
    deals = load_json_data("historical_deals.json")
    filtered = []
    for d in deals:
        if stage and d.get("stage", "").lower() != stage.lower():
            continue
        if outcome and d.get("outcome", "").lower() != outcome.lower():
            continue
        if min_size is not None and d.get("deal_size", 0) < min_size:
            continue
        health = calculate_deal_health(d)
        filtered.append({**d, **health})
    return filtered


@app.get("/api/deals/live")
def get_live_deals(
    stage: Optional[str] = None,
    risk_level: Optional[str] = None,
    min_size: Optional[float] = None
):
    filtered = []
    for processed in PROCESSED_LIVE_DEALS:
        if stage and processed.get("stage", "").lower() != stage.lower():
            continue
        if min_size is not None and processed.get("deal_size", 0) < min_size:
            continue
        if risk_level:
            clean_filter = risk_level.lower().replace("risk", "").strip()
            clean_category = processed["risk_category"].lower().replace("risk", "").strip()
            if clean_filter != clean_category:
                continue

        filtered.append(processed)
        
    return {
        "last_refreshed": LAST_REFRESHED_ISO,
        "deals_count": len(filtered),
        "deals": filtered
    }


@app.get("/api/metrics")
def get_pipeline_metrics():
    historical = load_json_data("historical_deals.json")
    
    # Deduplicate active live deals by unique deal_id
    seen_ids = set()
    active_live_deals = []
    for d in PROCESSED_LIVE_DEALS:
        deal_id = d.get("deal_id")
        outcome = str(d.get("outcome", "in_progress")).lower()
        if deal_id and deal_id not in seen_ids and outcome in ["in_progress", "active", "open"]:
            seen_ids.add(deal_id)
            active_live_deals.append(d)
    
    total_live_value = sum(d.get("deal_size", 0) for d in active_live_deals)
    weighted_pipeline_value = sum(d.get("deal_size", 0) * d["win_probability"] for d in active_live_deals)
    
    high_risk_deals = [d for d in active_live_deals if d.get("risk_category") == "High Risk"]
    med_risk_deals = [d for d in active_live_deals if d.get("risk_category") == "Medium Risk"]
    low_risk_deals = [d for d in active_live_deals if d.get("risk_category") == "Low Risk"]

    high_risk_value = sum(d.get("deal_size", 0) for d in high_risk_deals)
    medium_risk_value = sum(d.get("deal_size", 0) for d in med_risk_deals)
    low_risk_value = sum(d.get("deal_size", 0) for d in low_risk_deals)

    # Pipeline at risk includes High + Medium Risk active deals (risk_score >= 35.0)
    value_at_risk = high_risk_value + medium_risk_value

    avg_health_score = round(sum(d["health_score"] for d in active_live_deals) / max(1, len(active_live_deals)), 1)
    
    won_count = sum(1 for d in historical if d.get("outcome") == "won")
    historical_win_rate = round((won_count / max(1, len(historical))) * 100, 1)

    return {
        "live_deals_count": len(active_live_deals),
        "total_pipeline_value": total_live_value,
        "weighted_pipeline_value": round(weighted_pipeline_value, 2),
        "value_at_risk": round(value_at_risk, 2),
        "high_risk_value": round(high_risk_value, 2),
        "medium_risk_value": round(medium_risk_value, 2),
        "low_risk_value": round(low_risk_value, 2),
        "avg_health_score": avg_health_score,
        "risk_breakdown": {
            "high": len(high_risk_deals),
            "medium": len(med_risk_deals),
            "low": len(low_risk_deals)
        },
        "historical_total": len(historical),
        "historical_win_rate": historical_win_rate,
        "last_refreshed": LAST_REFRESHED_ISO
    }


@app.post("/api/extract-features", response_model=EngagementFeatureVector)
def extract_deal_features(deal_record: Dict[str, Any]):
    return extract_engagement_features(deal_record)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def read_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return {"message": "DealIQ API is running."}
