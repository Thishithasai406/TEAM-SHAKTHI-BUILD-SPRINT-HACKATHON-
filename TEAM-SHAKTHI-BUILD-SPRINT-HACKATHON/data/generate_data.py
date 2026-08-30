import json
import csv
import random
import os
from pathlib import Path

# Ensure reproducible dataset generation
random.seed(42)

DATA_DIR = Path(__file__).parent

COMPANY_NAMES = [
    "Acme Corp", "Apex Global", "Aether Systems", "BlueWave Tech", "Beacon Dynamics",
    "CloudScale", "CyberShield", "DataCore Inc", "Delta Analytics", "Evolve Logic",
    "Forward Media", "Frontier AI", "Genesis Cloud", "Hyperion Systems", "InfiniTech",
    "IronClad Security", "Krypton Labs", "Luminary Data", "Matrix Networks", "Nexus Enterprise",
    "OmniSoft", "Optima Solutions", "Pinnacle Networks", "Quantum Dynamics", "Radiant Systems",
    "ScaleGrid", "Sentinels Security", "Stratum Software", "Titan Technologies", "Vanguard Cloud",
    "Vertex Systems", "Velocity Networks", "Zenith Enterprise", "ZeroDay Defense", "Aura Insights"
]

STAGES = ["Discovery", "Qualification", "Proposal", "Negotiation", "Closing"]
SENTIMENT_TRENDS = ["improving", "stable", "declining"]

def generate_deal(deal_id, is_live=False):
    company = random.choice(COMPANY_NAMES)
    deal_name = f"{company} - Enterprise Platform Renewal & Expansion" if random.random() > 0.5 else f"{company} - Digital Transformation Deal"
    
    # Deal size ($20,000 to $1,500,000)
    deal_size = round(random.uniform(20000, 1500000), -3)
    
    # Realistic correlated features
    stakeholder_count = random.randint(1, 12)
    days_in_stage = random.randint(3, 180)
    response_latency_hrs = round(random.uniform(0.5, 72.0), 1)
    sentiment_trend = random.choice(SENTIMENT_TRENDS)
    competitor_mentions = random.randint(0, 6)
    scope_change_flags = random.randint(0, 4)
    stage = random.choice(STAGES) if is_live else "Closed"

    # Compute a latent "health score" to establish realistic outcome correlations
    health_score = 50.0
    
    # Health penalties/bonuses
    if days_in_stage > 60:
        health_score -= (days_in_stage - 60) * 0.4
    if response_latency_hrs > 24:
        health_score -= (response_latency_hrs - 24) * 0.8
    if sentiment_trend == "declining":
        health_score -= 25
    elif sentiment_trend == "improving":
        health_score += 15
    if competitor_mentions > 2:
        health_score -= competitor_mentions * 6
    if scope_change_flags > 2:
        health_score -= scope_change_flags * 5
    if stakeholder_count >= 4:
        health_score += 10
    else:
        health_score -= 10
        
    # Deal size complexity adjustment
    if deal_size > 500000 and stakeholder_count < 3:
        health_score -= 15

    # Determine outcome for historical data based on health score with noise
    if is_live:
        outcome = "in_progress"
    else:
        # Probabilities derived from health score
        if health_score > 40:
            prob_won = min(0.9, 0.4 + (health_score - 40) * 0.01)
            prob_stalled = (1 - prob_won) * 0.6
        elif health_score > 10:
            prob_won = 0.2
            prob_stalled = 0.5
        else:
            prob_won = 0.05
            prob_stalled = 0.35
            
        rand_val = random.random()
        if rand_val < prob_won:
            outcome = "won"
        elif rand_val < prob_won + prob_stalled:
            outcome = "stalled"
        else:
            outcome = "lost"

    deal = {
        "deal_id": f"DEAL-{deal_id:04d}",
        "company_name": company,
        "deal_name": deal_name,
        "stage": stage,
        "deal_size": deal_size,
        "days_in_stage": days_in_stage,
        "response_latency_hrs": response_latency_hrs,
        "sentiment_trend": sentiment_trend,
        "stakeholder_count": stakeholder_count,
        "competitor_mentions": competitor_mentions,
        "scope_change_flags": scope_change_flags,
        "outcome": outcome
    }
    return deal

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate 550 historical deals and 100 live deals (Total 650 > 500 requirement)
    historical_deals = [generate_deal(i + 1, is_live=False) for i in range(550)]
    live_deals = [generate_deal(i + 551, is_live=True) for i in range(100)]
    
    # Save historical deals (labeled)
    with open(DATA_DIR / "historical_deals.json", "w") as f:
        json.dump(historical_deals, f, indent=2)
        
    with open(DATA_DIR / "historical_deals.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=historical_deals[0].keys())
        writer.writeheader()
        writer.writerows(historical_deals)

    # Save live deals (outcome hidden / set to in_progress)
    with open(DATA_DIR / "live_deals.json", "w") as f:
        json.dump(live_deals, f, indent=2)

    with open(DATA_DIR / "live_deals.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=live_deals[0].keys())
        writer.writeheader()
        writer.writerows(live_deals)

    print(f"Successfully generated {len(historical_deals)} historical deals and {len(live_deals)} live deals in {DATA_DIR}.")

if __name__ == "__main__":
    main()
