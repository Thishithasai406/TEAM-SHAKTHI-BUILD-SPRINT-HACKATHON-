import json
import numpy as np
import scipy.stats as stats
from main import load_json_data

hist = load_json_data('historical_deals.json')
WON = [d for d in hist if d['outcome'] == 'won']
FAIL = [d for d in hist if d['outcome'] in ['stalled', 'lost']]

features = ['deal_size', 'days_in_stage', 'response_latency_hrs', 'stakeholder_count', 'competitor_mentions', 'scope_change_flags']

print('STATISTICAL DIFFERENCE TESTS (WON vs STALLED/LOST):')
print(f'{"Feature":<22} | {"WON Mean":<10} | {"FAIL Mean":<10} | {"t-statistic":<12} | {"p-value":<10}')
print('-' * 70)

for f in features:
    w_vals = [d[f] for d in WON]
    f_vals = [d[f] for d in FAIL]
    t_stat, p_val = stats.ttest_ind(w_vals, f_vals, equal_var=False)
    print(f'{f:<22} | {np.mean(w_vals):<10.1f} | {np.mean(f_vals):<10.1f} | {t_stat:<12.4f} | {p_val:<10.4e}')

# Chi-square for sentiment
cont_table = [
    [[d['sentiment_trend'] for d in WON].count('improving'), [d['sentiment_trend'] for d in FAIL].count('improving')],
    [[d['sentiment_trend'] for d in WON].count('stable'), [d['sentiment_trend'] for d in FAIL].count('stable')],
    [[d['sentiment_trend'] for d in WON].count('declining'), [d['sentiment_trend'] for d in FAIL].count('declining')]
]
chi2, p_val_sent, dof, ex = stats.chi2_contingency(cont_table)
print(f'{"sentiment_trend":<22} | {"-":<10} | {"-":<10} | {chi2:<12.4f} | {p_val_sent:<10.4e}')
