import re
import os
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class EngagementFeatureVector(BaseModel):
    deal_id: Optional[str] = None
    response_latency_hrs: float = Field(..., description="Average response latency in hours")
    sentiment_trend: str = Field(..., description="Extracted sentiment trend: 'improving', 'stable', or 'declining'")
    sentiment_score: float = Field(..., description="Numerical sentiment score (-1.0 to +1.0)")
    stakeholder_count: int = Field(..., description="Total count of unique stakeholders identified")
    stakeholders_found: List[str] = Field(default_factory=list, description="List of stakeholder names/roles identified")
    competitor_mentions: int = Field(..., description="Count of competitor mentions identified")
    competitors_found: List[str] = Field(default_factory=list, description="List of competitor names found in text")
    scope_change_flags: int = Field(..., description="Count of scope-change language indicators found")
    scope_change_phrases: List[str] = Field(default_factory=list, description="Matched scope change phrases")
    extractor_used: str = Field(default="heuristic", description="Method used for extraction: 'llm' or 'heuristic'")


KNOWN_COMPETITION = [
    "salesforce", "hubspot", "gong", "clari", "chorus", "outreach", 
    "salesloft", "microsoft", "dynamics", "oracle", "sap", "competitor x", "vendor b"
]

STAKEHOLDER_ROLES = [
    "vp", "director", "cto", "cio", "cfo", "ceo", "ciso", "head of", 
    "manager", "procurement", "legal", "decision maker", "champion", "buyer", "executive"
]

SCOPE_CHANGE_PATTERNS = [
    r"\bscope\b", r"\brequirement[s]?\b", r"\badd-on[s]?\b", r"\bchange request\b",
    r"\bcustomization[s]?\b", r"\badditional feature[s]?\b", r"\bscale back\b",
    r"\bexpanded scope\b", r"\breduce scope\b", r"\bnew module[s]?\b", r"\btimeline shift\b"
]

POSITIVE_SENTIMENT_WORDS = {
    "great", "excellent", "excited", "impressed", "approve", "agreed", "moving forward", 
    "love", "positive", "aligned", "value", "perfect", "good", "promising", "clear"
}

NEGATIVE_SENTIMENT_WORDS = {
    "delay", "concern", "risk", "expensive", "issue", "problem", "blocker", 
    "cancel", "frustrated", "hesitant", "doubt", "competitor", "slow", "unclear", "hold"
}


def analyze_text_llm(text: str) -> Optional[Dict[str, Any]]:
    """Uses OpenAI LLM to analyze conversation text and extract structured engagement features."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = f"""
Analyze the following enterprise sales call transcript or email string and extract structured engagement features:

Text to analyze:
\"\"\"
{text}
\"\"\"

Return JSON with exactly these keys:
- "sentiment_score": float between -1.0 (extremely negative) and +1.0 (extremely positive)
- "sentiment_trend": string ("improving", "stable", or "declining")
- "stakeholders": list of stakeholder names or role titles identified
- "competitors": list of competitor company names mentioned
- "scope_phrases": list of scope-change phrases or flags mentioned
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a sales engagement NLP analyzer. Output strict JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        content = response.choices[0].message.content
        data = json.loads(content)
        data["extractor_used"] = "llm"
        return data

    except Exception as e:
        print(f"[LLM Extractor Fallback] Error invoking OpenAI LLM: {e}")
        return None


def analyze_text_nlp(text: str) -> Dict[str, Any]:
    """Applies rule-based NLP extraction on email/transcript text as fallback or primary engine."""
    if not text:
        return {
            "sentiment_score": 0.0,
            "sentiment_trend": "stable",
            "stakeholders": [],
            "competitors": [],
            "scope_phrases": [],
            "extractor_used": "heuristic"
        }

    lower_text = text.lower()

    # 1. Competitor Mentions
    found_competitors = []
    for comp in KNOWN_COMPETITION:
        if re.search(r'\b' + re.escape(comp) + r'\b', lower_text):
            found_competitors.append(comp.capitalize())

    # 2. Stakeholders Extraction
    found_stakeholders = set()
    for role in STAKEHOLDER_ROLES:
        matches = re.findall(r'(\b[A-Z][a-z]+\s+)?\b' + re.escape(role) + r'\b(\s+[A-Z][a-z]+)?', text, re.IGNORECASE)
        for match in matches:
            full = " ".join([m.strip() for m in match if m.strip()])
            if full:
                found_stakeholders.add(full.title())
    
    names = re.findall(r'(?:from|attending|joined|cc|to):\s*([A-Z][a-z]+\s+[A-Z][a-z]+)', text, re.IGNORECASE)
    for n in names:
        found_stakeholders.add(n.title())

    # 3. Scope Change Language
    found_scope_phrases = []
    for pattern in SCOPE_CHANGE_PATTERNS:
        matches = re.findall(pattern, lower_text)
        if matches:
            found_scope_phrases.extend(matches)

    # 4. Lexicon Sentiment Analysis & Trend
    words = re.findall(r'\w+', lower_text)
    pos_count = sum(1 for w in words if w in POSITIVE_SENTIMENT_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_SENTIMENT_WORDS)
    
    total_sentiment_words = pos_count + neg_count
    if total_sentiment_words > 0:
        sentiment_score = (pos_count - neg_count) / total_sentiment_words
    else:
        sentiment_score = 0.0

    if sentiment_score > 0.2:
        sentiment_trend = "improving"
    elif sentiment_score < -0.2:
        sentiment_trend = "declining"
    else:
        sentiment_trend = "stable"

    return {
        "sentiment_score": round(sentiment_score, 2),
        "sentiment_trend": sentiment_trend,
        "stakeholders": list(found_stakeholders),
        "competitors": found_competitors,
        "scope_phrases": found_scope_phrases,
        "extractor_used": "heuristic"
    }


def extract_engagement_features(raw_deal_record: Dict[str, Any], use_llm: bool = True) -> EngagementFeatureVector:
    """
    Extracts engagement feature vector from a raw deal record containing structured
    fields and optional transcript/email text (`text` or `transcript` key).
    Attempts real LLM call if enabled and API key is present; falls back seamlessly to rule-based NLP.
    """
    deal_id = raw_deal_record.get("deal_id")
    transcript_text = raw_deal_record.get("transcript") or raw_deal_record.get("email_text") or raw_deal_record.get("text", "")
    
    nlp_results = None
    if use_llm and transcript_text and len(transcript_text.strip()) > 10:
        nlp_results = analyze_text_llm(transcript_text)

    if not nlp_results:
        nlp_results = analyze_text_nlp(transcript_text)

    response_latency_hrs = float(
        raw_deal_record.get("response_latency_hrs", raw_deal_record.get("response_latency", 24.0))
    )

    struct_competitors = raw_deal_record.get("competitor_mentions", 0)
    if isinstance(struct_competitors, int):
        competitor_count = max(struct_competitors, len(nlp_results["competitors"]))
    else:
        competitor_count = len(nlp_results["competitors"])

    struct_stakeholders = raw_deal_record.get("stakeholder_count", 0)
    text_stakeholder_count = len(nlp_results["stakeholders"])
    stakeholder_count = max(struct_stakeholders, text_stakeholder_count, 1)

    struct_scope_flags = raw_deal_record.get("scope_change_flags", 0)
    text_scope_count = len(nlp_results["scope_phrases"])
    scope_change_flags = max(struct_scope_flags, text_scope_count)

    sentiment_trend = raw_deal_record.get("sentiment_trend") or nlp_results["sentiment_trend"]

    return EngagementFeatureVector(
        deal_id=deal_id,
        response_latency_hrs=response_latency_hrs,
        sentiment_trend=sentiment_trend,
        sentiment_score=float(nlp_results.get("sentiment_score", 0.0)),
        stakeholder_count=stakeholder_count,
        stakeholders_found=nlp_results.get("stakeholders", []),
        competitor_mentions=competitor_count,
        competitors_found=nlp_results.get("competitors", []),
        scope_change_flags=scope_change_flags,
        scope_change_phrases=nlp_results.get("scope_phrases", []),
        extractor_used=nlp_results.get("extractor_used", "heuristic")
    )
