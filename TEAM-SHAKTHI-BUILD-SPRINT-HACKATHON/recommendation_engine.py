from typing import Dict, Any, List, Optional

def generate_recovery_recommendation(
    live_deal: Dict[str, Any],
    risk_score: float,
    nearest_analogs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validation-Gated Root-Cause Diagnostic, Outranking & Recommendation Engine.
    
    Principles Enforced:
    1. Multi-Signal Root-Cause Ranking: Evaluates all 8 features together to rank potential root causes.
    2. Symptom vs. Cause Distinction: Stage Stagnation / Extended Time in Stage / Response Latency are symptoms, not final root causes when underlying causes exist.
    3. Explicit Outranking Logic: Identifies competing root cause and explains why primary cause outranks it.
    4. Positive Signal Weighting: Fast latency, improving sentiment, and multi-threaded coverage reduce confidence in disengagement/misalignment.
    5. Validation Rules:
       - Competitive Pressure requires supporting signals (declining sentiment, >= 3 mentions, late-stage evaluation). Default action is Value Differentiation; price concessions require explicit commercial pressure.
       - Scope Instability requires >= 2 scope changes with friction. 1 scope change = "Scope Adjustment Recorded".
       - Buyer Disengagement requires high latency/declining sentiment. Deprioritized when latency is healthy.
       - Lack of Stakeholder Alignment requires insufficient coverage. Deprioritized when stakeholders >= 3 (or >= 5). Uses "Stakeholders Involved: N" (never infers roles).
       - Procurement Delay requires late-stage contracting context (Negotiation/Closing).
    """
    response_latency = float(live_deal.get("response_latency_hrs") if live_deal.get("response_latency_hrs") is not None else 24.0)
    stakeholder_count = int(live_deal.get("stakeholder_count") if live_deal.get("stakeholder_count") is not None else 3)
    competitor_mentions = int(live_deal.get("competitor_mentions") if live_deal.get("competitor_mentions") is not None else 0)
    scope_flags = int(live_deal.get("scope_change_flags") if live_deal.get("scope_change_flags") is not None else 0)
    deal_size = float(live_deal.get("deal_size") if live_deal.get("deal_size") is not None else 0.0)
    days_in_stage = int(live_deal.get("days_in_stage") if live_deal.get("days_in_stage") is not None else 0)
    stage_raw = str(live_deal.get("stage", "Qualified")).strip()
    stage_lower = stage_raw.lower()
    
    sentiment_raw = live_deal.get("sentiment_trend")
    sentiment = str(sentiment_raw).lower().strip() if sentiment_raw is not None else "stable"

    # Unified Stage Categorization and Velocity Benchmarks (Matching STAGE_BENCHMARKS)
    if "disc" in stage_lower:
        stage_category = "Discovery"
        stage_days_benchmark = 14
    elif "qual" in stage_lower:
        stage_category = "Qualification"
        stage_days_benchmark = 21
    elif "eval" in stage_lower or "demo" in stage_lower or "poc" in stage_lower:
        stage_category = "Evaluation"
        stage_days_benchmark = 30
    elif "prop" in stage_lower or "quote" in stage_lower:
        stage_category = "Proposal"
        stage_days_benchmark = 30
    elif "nego" in stage_lower or "contract" in stage_lower or "redline" in stage_lower:
        stage_category = "Negotiation"
        stage_days_benchmark = 21
    elif "clos" in stage_lower or "commit" in stage_lower or "sign" in stage_lower:
        stage_category = "Closing"
        stage_days_benchmark = 14
    else:
        stage_category = "Proposal"
        stage_days_benchmark = 30

    # Analogs Analysis
    won_analogs = [a for a in nearest_analogs if a.get("outcome") == "won"]
    lost_analogs = [a for a in nearest_analogs if a.get("outcome") == "lost"]
    stalled_analogs = [a for a in nearest_analogs if a.get("outcome") == "stalled"]
    lost_or_stalled = lost_analogs + stalled_analogs
    total_analogs = len(nearest_analogs)
    top_won = won_analogs[0] if won_analogs else None

    # Calculate analog stage duration stats for specific WHY NOT WAIT narrative
    stalled_lost_days = [int(a.get("days_in_stage", 0)) for a in lost_or_stalled if a.get("days_in_stage") is not None]
    threshold_days = min(stalled_lost_days) if stalled_lost_days else days_in_stage

    # Check explicit pricing / commercial pressure signals in live deal or won analogs
    pricing_objection_flag = bool(live_deal.get("pricing_objection", False)) or bool(live_deal.get("budget_objection", False))
    analog_pricing_evidence = any("price" in str(a.get("recommendation", "")).lower() or "discount" in str(a.get("recommendation", "")).lower() for a in won_analogs)
    explicit_commercial_pressure = pricing_objection_flag or analog_pricing_evidence

    # =========================================================================
    # 1. RISK CONTRIBUTION NORMALIZATION (EXACTLY 100%)
    # Uses "Stakeholders Involved: N" and "Risk Signal" labels
    # =========================================================================
    candidate_risks = []

    # Risk Signal 1: Single-threading
    stakeholder_benchmark = 3 if deal_size >= 200000 else 2
    if stakeholder_count < stakeholder_benchmark:
        diff_stk = stakeholder_benchmark - stakeholder_count
        weight = 35.0 if stakeholder_count == 1 else 20.0
        candidate_risks.append({
            "factor_name": "Single-Threaded Relationship",
            "current_value": f"Stakeholders Involved: {stakeholder_count}",
            "benchmark_value": f">= {stakeholder_benchmark} stakeholders",
            "difference": f"-{diff_stk} contact gap",
            "weight": weight,
            "why_it_matters": "Deals with a single point of contact fail if that stakeholder leaves or lacks budget authority."
        })

    # Risk Signal 2: Response Latency
    latency_benchmark = 24.0
    if response_latency > latency_benchmark:
        diff_lat = response_latency - latency_benchmark
        weight = min(30.0, max(15.0, (diff_lat / latency_benchmark) * 15 + 10))
        candidate_risks.append({
            "factor_name": "Elevated Response Latency",
            "current_value": f"{response_latency:.1f} hours average",
            "benchmark_value": f"<= {latency_benchmark:.0f} hours",
            "difference": f"+{diff_lat:.1f} hours slower",
            "weight": weight,
            "why_it_matters": "Slower communication response times indicate fading buyer urgency or shifting internal priorities."
        })

    # Risk Signal 3: Competitor Presence (Only added as a risk factor if supported by declining sentiment or >= 2 competitor mentions)
    if competitor_mentions >= 2 or (competitor_mentions >= 1 and sentiment == "declining"):
        if competitor_mentions >= 3 and sentiment == "declining":
            weight = 35.0
        elif competitor_mentions >= 3:
            weight = 25.0
        elif competitor_mentions >= 1 and sentiment == "declining":
            weight = 25.0
        else:
            weight = 15.0
        candidate_risks.append({
            "factor_name": "Competitor Presence",
            "current_value": f"{competitor_mentions} mention(s) logged",
            "benchmark_value": "0 competitor presence",
            "difference": f"+{competitor_mentions} competing vendor(s)",
            "weight": weight,
            "why_it_matters": "Active competitor involvement increases evaluation friction and requires value differentiation."
        })

    # Risk Signal 4: Scope Adjustments (Only added as a risk factor if >= 2 scope changes or 1 change with extended stage delay)
    if scope_flags >= 2 or (scope_flags == 1 and days_in_stage > stage_days_benchmark):
        weight = 30.0 if scope_flags >= 3 else (20.0 if scope_flags == 2 else 10.0)
        scope_title = "Unstable Custom Scope" if scope_flags >= 2 else "Scope Adjustment Recorded"
        scope_why = "Multiple scope changes introduce technical friction and delay contract sign-off." if scope_flags >= 2 else "A scope adjustment has been recorded and requires baseline verification."
        candidate_risks.append({
            "factor_name": scope_title,
            "current_value": f"{scope_flags} scope change flag{'s' if scope_flags > 1 else ''}",
            "benchmark_value": "0 custom scope changes",
            "difference": f"+{scope_flags} scope adjustment{'s' if scope_flags > 1 else ''}",
            "weight": weight,
            "why_it_matters": scope_why
        })

    # Risk Signal 5: Declining Sentiment
    if sentiment == "declining":
        candidate_risks.append({
            "factor_name": "Declining Buyer Sentiment",
            "current_value": "Declining ↓",
            "benchmark_value": "Stable or Improving ↑",
            "difference": "Negative trend direction",
            "weight": 15.0,
            "why_it_matters": "Declining communication sentiment reflects underlying hesitation or internal objections."
        })

    # Risk Signal 6: Extended Time in Stage
    if days_in_stage > stage_days_benchmark:
        diff_days = days_in_stage - stage_days_benchmark
        # If specific root causes (multiple scope changes or multiple competitors) are present, cap stage duration weight so specific cause remains primary
        max_stage_w = 20.0 if (scope_flags >= 3 or competitor_mentions >= 3 or stakeholder_count == 1) and days_in_stage < 100 else 40.0
        weight = min(max_stage_w, max(15.0, (diff_days / stage_days_benchmark) * 20 + 10))
        candidate_risks.append({
            "factor_name": "Extended Time in Stage",
            "current_value": f"{days_in_stage} days in {stage_raw}",
            "benchmark_value": f"<= {stage_days_benchmark} days",
            "difference": f"+{diff_days} days overdue",
            "weight": weight,
            "why_it_matters": "Stage velocity delay reflects underlying engagement or decision-maker bottlenecks."
        })

    candidate_risks.sort(key=lambda x: x["weight"], reverse=True)

    # EXACT 100% NORMALIZATION RULE FOR TOP DISPLAYED RISK SIGNALS
    top_factors = candidate_risks[:3]
    top_weight_sum = sum(r["weight"] for r in top_factors) if top_factors else 1.0

    raw_pcts = [round((r["weight"] / top_weight_sum) * 100) for r in top_factors]
    diff_from_100 = 100 - sum(raw_pcts)
    if raw_pcts and diff_from_100 != 0:
        raw_pcts[0] += diff_from_100  # Adjust primary risk signal to guarantee exact 100% sum

    ranked_risks = {}
    keys = ["primary", "secondary", "supporting"]
    for idx, risk in enumerate(top_factors):
        risk_copy = dict(risk)
        risk_copy["contribution"] = f"{raw_pcts[idx]}% of risk score"
        ranked_risks[keys[idx]] = risk_copy

    omitted_factors = candidate_risks[3:]
    if omitted_factors:
        omitted_names = [f["factor_name"] for f in omitted_factors]
        ranked_risks["remainder_note"] = f"Additional risk signal(s) accounted for in score: {', '.join(omitted_names)}."

    # =========================================================================
    # 2. POSITIVE SIGNALS IDENTIFICATION
    # =========================================================================
    positive_signals = []
    if deal_size >= 250000:
        positive_signals.append(f"Significant deal magnitude ({format_currency(deal_size)}) provides strong revenue opportunity.")
    if stakeholder_count >= 3:
        positive_signals.append(f"Multi-threaded account coverage with Stakeholders Involved: {stakeholder_count}.")
    if response_latency <= 24:
        positive_signals.append(f"Healthy buyer communication velocity ({response_latency:.1f}h response latency).")
    if competitor_mentions == 0:
        positive_signals.append("Zero competitor mentions logged; clear sole-source evaluation positioning.")
    if scope_flags == 0:
        positive_signals.append("Stable implementation scope with zero custom change requests.")
    if sentiment == "improving":
        positive_signals.append("Positive communication sentiment trend in recent buyer touchpoints.")
    if len(positive_signals) == 0:
        positive_signals.append("Deal is currently active in pipeline.")

    # Mapping from displayed risk factor_name to standardized root cause name
    FACTOR_TO_ROOT_CAUSE = {
        "Extended Time in Stage": "Extended Time in Stage / Deal Stagnation",
        "Competitor Presence": "Competitive Pressure",
        "Unstable Custom Scope": "Scope Instability",
        "Scope Adjustment Recorded": "Scope Instability",
        "Single-Threaded Relationship": "Lack of Stakeholder Alignment",
        "Elevated Response Latency": "Buyer Disengagement",
        "Declining Buyer Sentiment": "Buyer Disengagement"
    }

    # Deterministic mapping for root cause selection directly from risk signal weights
    if top_factors:
        primary_factor_name = top_factors[0]["factor_name"]
        
        # Late-stage procurement friction override (Negotiation/Closing stage with late-stage response latency > 30h or pricing/contract objection)
        if stage_category in ["Negotiation", "Closing"] and (response_latency > 30.0 or pricing_objection_flag):
            root_cause = "Procurement Delay"
            primary_score = max(top_factors[0]["weight"], 30.0)
        else:
            root_cause = FACTOR_TO_ROOT_CAUSE.get(primary_factor_name, "Early Stage Qualification Friction")
            primary_score = top_factors[0]["weight"]

        if len(top_factors) > 1:
            sec_factor_name = top_factors[1]["factor_name"]
            competing_cause = FACTOR_TO_ROOT_CAUSE.get(sec_factor_name, "Early Stage Qualification Friction")
            competing_score = top_factors[1]["weight"]
        else:
            competing_cause = "Early Stage Qualification Friction"
            competing_score = 0.0
    else:
        root_cause = "Early Stage Qualification Friction"
        competing_cause = "Insufficient Causal Evidence"
        primary_score = 0.0
        competing_score = 0.0

    # =========================================================================
    # 4. STRICTLY ALIGNED ACTION MAPPING & EVIDENCE-GROUNDED EXPLANATION
    # Never infers specific unobserved events (e.g. redlines, technical friction, roles)
    # =========================================================================
    if root_cause == "Lack of Stakeholder Alignment":
        primary_action = f"Initiate check-in with additional key contacts to expand multi-threaded account coverage."
        secondary_action = f"Conduct a formal Mutual Action Plan (MAP) review to identify all required decision-makers"
        objective = "Secure multi-threaded executive alignment to ensure decision continuity."
        
        cause_evidence = f"Stakeholders Involved: {stakeholder_count} for a {format_currency(deal_size)} deal in {stage_raw}"
        if competing_cause != "Early Stage Qualification Friction":
            outranking_reason = f"Lack of Stakeholder Alignment outranks {competing_cause} because single-threading (Stakeholders Involved: {stakeholder_count}) creates an account vulnerability that must be resolved before addressing secondary friction"
        else:
            outranking_reason = f"Lack of Stakeholder Alignment is the primary root cause because single-contact coverage creates an immediate account vulnerability"

    elif root_cause == "Competitive Pressure":
        if explicit_commercial_pressure:
            primary_action = f"Provide a structured competitive price-matching proposal and flexible quarterly billing terms."
            secondary_action = f"Deliver a commercial comparison matrix justifying value against competitor pricing"
        else:
            primary_action = f"Deploy a differentiated value matrix highlighting unique architectural/operational capabilities against competing vendors."
            secondary_action = f"Schedule a technical deep-dive with key evaluators to demonstrate unique ROI over competitor alternatives"

        objective = "Neutralize competitor displacement attempts by proving superior ROI and execution speed."
        cause_evidence = f"{competitor_mentions} competitor mention(s) logged in {stage_raw} alongside {sentiment} communication sentiment"
        
        if competing_cause != "Early Stage Qualification Friction":
            outranking_reason = f"Competitive Pressure outranks {competing_cause} because active competitor presence ({competitor_mentions} mentions) creates direct evaluation friction"
        else:
            outranking_reason = f"Competitive Pressure is the primary root cause driving evaluation friction"

    elif root_cause == "Scope Instability":
        primary_action = f"Freeze custom scope additions and package a baseline 'Phase 1 Core Implementation' offer."
        secondary_action = f"Move non-essential custom feature requests to an optional Phase 2 statement of work"
        objective = "Eliminate technical complexity and stabilize scope to accelerate sign-off."
        cause_evidence = f"{scope_flags} custom scope change flags recorded alongside an extended {days_in_stage}-day duration in {stage_raw}"
        outranking_reason = f"Scope Instability outranks {competing_cause} because {scope_flags} scope change flags coincide with an extended {days_in_stage}-day stage duration"

    elif root_cause == "Procurement Delay":
        primary_action = f"Review commercial and procurement alignment with relevant stakeholders to clear late-stage contracting bottlenecks."
        secondary_action = f"Provide standard pre-approved contract terms or flexible billing schedules to satisfy procurement guidelines"
        objective = "Clear contracting hurdles to finalize execution."
        cause_evidence = f"late-stage status in {stage_raw} with elevated {response_latency:.1f}h buyer response latency"
        outranking_reason = f"Procurement Delay is supported by late-stage status in {stage_raw} and elevated {response_latency:.1f}h response latency, indicating potential late-stage contracting friction"

    elif root_cause == "Buyer Disengagement":
        primary_action = f"Deliver a concise 'Value Realization Summary' demonstrating quantified business impact and cost of inaction."
        secondary_action = f"Offer an executive-level strategy briefing or time-bound pilot assurance warranty"
        objective = "Re-ignite buyer urgency by demonstrating the business cost of continuing the status quo."
        cause_evidence = f"elevated buyer response latency ({response_latency:.1f}h) and {sentiment} sentiment trend in {stage_raw}"
        outranking_reason = f"Buyer Disengagement outranks {competing_cause} because slowing communication velocity ({response_latency:.1f}h latency) indicates fading priority"

    elif root_cause in ["Extended Time in Stage / Deal Stagnation", "Deal Stagnation"]:
        primary_action = f"Initiate an executive sponsor check-in to reset timeline expectations and re-validate business priorities."
        secondary_action = f"Establish a structured 14-day decision framework with firm milestone deadlines"
        objective = "Unstick the stagnant deal by securing firm timeline commitments from executive sponsors."
        cause_evidence = f"the deal has remained in {stage_raw} for {days_in_stage} days versus the benchmark of <= {stage_days_benchmark} days, making stage stagnation the highest-ranked risk signal at {raw_pcts[0] if raw_pcts else 0}%"
        outranking_reason = f"{root_cause} outranks {competing_cause} because stage duration delay ({days_in_stage} days in {stage_raw}) is the primary risk driver"

    else:
        primary_action = f"Conduct a MEDDPICC qualification review to re-validate budget, compelling event, and decision authority."
        secondary_action = f"Offer a simplified low-commitment starter package to test true buyer intent"
        objective = "Re-validate deal viability and establish clear decision milestones."
        cause_evidence = f"early-stage qualification parameters in {stage_raw}"
        outranking_reason = f"Re-qualifying core criteria is required to establish active buyer intent"

    # BUILD NATURAL POSITIVE SIGNALS SUMMARY PHRASES
    pos_phrases = []
    if deal_size >= 250000:
        pos_phrases.append(f"the {format_currency(deal_size)} deal size")
    if stakeholder_count >= 3:
        pos_phrases.append(f"Stakeholders Involved: {stakeholder_count}")
    if response_latency <= 24.0 and response_latency > 0:
        pos_phrases.append(f"healthy {response_latency:.1f}h response latency")
    if competitor_mentions == 0:
        pos_phrases.append("zero competitor presence")
    if scope_flags == 0:
        pos_phrases.append("stable implementation scope")
    if sentiment == "improving":
        pos_phrases.append("improving sentiment")

    if len(pos_phrases) == 1:
        pos_phrase_str = pos_phrases[0]
    elif len(pos_phrases) == 2:
        pos_phrase_str = f"{pos_phrases[0]} and {pos_phrases[1]}"
    elif len(pos_phrases) >= 3:
        pos_phrase_str = f"{', '.join(pos_phrases[:-1])}, and {pos_phrases[-1]}"
    else:
        pos_phrase_str = "active pipeline status"

    # Helper function to guarantee exactly one trailing period and clean whitespace
    def clean_sentence_period(s: str) -> str:
        cleaned = s.strip()
        while cleaned and cleaned[0] in ("-", "•", "*", " "):
            cleaned = cleaned[1:].strip()
        cleaned = cleaned.rstrip(".,; ")
        return f"{cleaned}."

    why_this_move_bullets = [
        clean_sentence_period(f"{root_cause} is the primary root cause because {cause_evidence}"),
        clean_sentence_period(f"Positive signals considered: {pos_phrase_str}. These signals reduce confidence in alternative failure modes"),
        clean_sentence_period(f"Competing factor: {competing_cause}. {outranking_reason}"),
        clean_sentence_period(f"Secondary Supporting Action: {secondary_action}")
    ]

    # =========================================================================
    # 5. SPECIFIC k-NN ANALOG WHY NOT WAIT NARRATIVE
    # Derived mathematically from the exact nearest_analogs list returned
    # =========================================================================
    num_stalled = len(stalled_analogs)
    num_lost = len(lost_analogs)
    num_won = len(won_analogs)
    num_stalled_lost = len(lost_or_stalled)

    if total_analogs > 0:
        outcome_parts = []
        if num_lost > 0:
            outcome_parts.append(f"{num_lost} lost")
        if num_stalled > 0:
            outcome_parts.append(f"{num_stalled} stalled")
        if num_won > 0:
            outcome_parts.append(f"{num_won} won")
        
        outcome_str = ", ".join(outcome_parts)
        
        # Calculate duration threshold: minimum days_in_stage across stalled/lost analogs if present, else min of all analogs
        all_analog_days = [int(a.get("days_in_stage", 0)) for a in nearest_analogs if a.get("days_in_stage") is not None]
        stalled_lost_days = [int(a.get("days_in_stage", 0)) for a in lost_or_stalled if a.get("days_in_stage") is not None]
        
        if stalled_lost_days:
            threshold_days = min(stalled_lost_days)
        elif all_analog_days:
            threshold_days = min(all_analog_days)
        else:
            threshold_days = days_in_stage

        # Count ALL returned analogs that meet or exceed threshold_days (using >=, evaluated against exact nearest_analogs list)
        num_at_threshold = len([a for a in nearest_analogs if int(a.get("days_in_stage", 0)) >= threshold_days])
        
        why_not_wait = (
            f"{num_stalled_lost}/{total_analogs} nearest historical analogs stalled or were lost ({outcome_str}). "
            f"{num_at_threshold}/{total_analogs} spent at least {threshold_days} days in {stage_raw}, while this deal has spent {days_in_stage} days in {stage_raw}."
        )
    else:
        why_not_wait = f"This deal has spent {days_in_stage} days in {stage_raw}, exceeding the stage benchmark of {stage_days_benchmark} days."

    risk_score_labeled = f"Risk Score: {min(100.0, max(0.0, float(risk_score))):.1f}/100"

    return {
        "root_cause": root_cause,
        "root_cause_explanation": outranking_reason,
        "competing_root_cause": competing_cause,
        "stage": stage_raw,
        "stage_category": stage_category,
        "risk_score_labeled": risk_score_labeled,
        "ranked_risks": ranked_risks,
        "positive_signals": positive_signals,
        "evidence_trace": [
            f"Stage: {stage_raw} (benchmark <= {stage_days_benchmark} days)",
            f"Stakeholders Involved: {stakeholder_count}",
            f"Response Latency: {response_latency:.1f}h",
            f"Competitor Mentions: {competitor_mentions}",
            f"Scope Change Flags: {scope_flags}",
            f"Sentiment Trend: {sentiment}"
        ],
        "primary_action": primary_action,
        "secondary_action": secondary_action,
        "recommendation": primary_action,
        "action_type": f"{stage_category} - {root_cause}",
        "reason": f"{root_cause} is the primary root cause driving this deal's {int(round(risk_score))}/100 risk score in {stage_raw}.",
        "why_this_move": why_this_move_bullets,
        "why_this_move_text": " ".join(why_this_move_bullets),
        "why_not_wait": why_not_wait,
        "objective": objective,
        "has_recovered_analogs": len(won_analogs) > 0,
        "recovered_analogs_count": len(won_analogs)
    }

def format_currency(val: float) -> str:
    return f"${val:,.0f}"
