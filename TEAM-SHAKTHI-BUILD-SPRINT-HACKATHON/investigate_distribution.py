import json
from main import calculate_deal_health, load_json_data

def main():
    historical = load_json_data('historical_deals.json')
    scores_by_outcome = {'won': [], 'stalled': [], 'lost': []}

    for d in historical:
        h = calculate_deal_health(d)
        outcome = d.get('outcome', '').lower()
        if outcome in scores_by_outcome:
            scores_by_outcome[outcome].append(h['health_score'])

    print('Historical Dataset Validation (550 Deals):')
    for outcome, scores in scores_by_outcome.items():
        avg = sum(scores) / len(scores) if scores else 0
        min_s = min(scores) if scores else 0
        max_s = max(scores) if scores else 0
        print(f'  {outcome.upper()} (n={len(scores)}): avg={avg:.1f}, min={min_s:.1f}, max={max_s:.1f}')

    # Breakdown by Health Score ranges
    ranges = {'High Health (>=70)': 0, 'Medium Health (45-69.9)': 0, 'Low Health (<45)': 0}
    range_outcomes = {r: {'won': 0, 'stalled': 0, 'lost': 0} for r in ranges}

    for d in historical:
        h = calculate_deal_health(d)
        score = h['health_score']
        outcome = d.get('outcome', '').lower()
        
        if score >= 70:
            r = 'High Health (>=70)'
        elif score >= 45:
            r = 'Medium Health (45-69.9)'
        else:
            r = 'Low Health (<45)'
            
        ranges[r] += 1
        if outcome in range_outcomes[r]:
            range_outcomes[r][outcome] += 1

    print('\nDistribution by Health Score Ranges:')
    for r, count in ranges.items():
        outcomes = range_outcomes[r]
        won_p = (outcomes['won'] / count * 100) if count else 0
        stalled_p = (outcomes['stalled'] / count * 100) if count else 0
        lost_p = (outcomes['lost'] / count * 100) if count else 0
        print(f'  {r}: total={count} | WON={outcomes["won"]} ({won_p:.1f}%), STALLED={outcomes["stalled"]} ({stalled_p:.1f}%), LOST={outcomes["lost"]} ({lost_p:.1f}%)')

if __name__ == '__main__':
    main()
