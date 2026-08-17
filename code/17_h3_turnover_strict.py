import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

# ============================================================
# Load inputs
# ============================================================
ov = pd.read_csv('data/processed_strict/overconfidence_index.csv',
                 dtype={'gvkey': str})
panel = pd.read_csv('data/processed_strict/regression_panel.csv',
                    dtype={'gvkey': str})
ceo_panel = pd.read_csv('data/processed_strict/execucomp_ceo_panel.csv',
                        dtype={'gvkey': str})
turnovers = pd.read_csv('data/processed_strict/turnover_events.csv',
                        dtype={'gvkey': str})
universe = pd.read_csv('data/processed_strict/sample_firms.csv',
                       comment='#', dtype={'gvkey': str})

print(f"Total turnover events recorded: {len(turnovers)}")

# ============================================================
# 1) Attach the CEO identity to each firm-year in the panel
#    so we can split overconfidence by CEO tenure.
# ============================================================
ceo_panel = ceo_panel.rename(columns={'year': 'fyear'})
ceo_yr = ceo_panel[['gvkey', 'fyear', 'execid']].drop_duplicates(
    subset=['gvkey', 'fyear'], keep='first'
)

ov = ov.merge(ceo_yr, on=['gvkey', 'fyear'], how='left')
panel = panel.merge(ceo_yr, on=['gvkey', 'fyear'], how='left')

# ============================================================
# 2) Build the turnover cross-section
#    For each turnover, compute:
#       deltaOverconf  = mean(overconfidence | new CEO years)
#                      - mean(overconfidence | old CEO years)
#       deltaLeverage  = mean(book_leverage | post-turnover years)
#                      - mean(book_leverage | pre-turnover years)
#       deltaDebtShare = same idea
#       deltaControls  = same idea
# ============================================================
records = []
for _, ev in turnovers.iterrows():
    gv = ev['gvkey']
    yr = int(ev['year'])  # year the new CEO appears in ANNCOMP
    new_execid = ev['execid']
    old_execid = ev['prev_execid']

    firm_ov = ov[ov['gvkey'] == gv]
    firm_panel = panel[panel['gvkey'] == gv]

    pre_ov  = firm_ov[firm_ov['execid'] == old_execid]['overconfidence']
    post_ov = firm_ov[firm_ov['execid'] == new_execid]['overconfidence']
    if len(pre_ov) == 0 or len(post_ov) == 0:
        continue

    pre_panel  = firm_panel[firm_panel['execid'] == old_execid]
    post_panel = firm_panel[firm_panel['execid'] == new_execid]
    if len(pre_panel) == 0 or len(post_panel) == 0:
        continue

    rec = {
        'gvkey': gv,
        'turnover_year': yr,
        'n_pre_calls':  len(pre_ov),
        'n_post_calls': len(post_ov),
        'delta_overconf':       post_ov.mean() - pre_ov.mean(),
        'delta_book_leverage':  (post_panel['book_leverage_t1'].mean()
                                  - pre_panel['book_leverage_t1'].mean()),
        'delta_debt_share':     (post_panel['debt_share_t1'].mean()
                                  - pre_panel['debt_share_t1'].mean()),
        'delta_size':           (post_panel['size'].mean()
                                  - pre_panel['size'].mean()),
        'delta_profitability':  (post_panel['profitability'].mean()
                                  - pre_panel['profitability'].mean()),
        'delta_mtb':            (post_panel['mtb'].mean()
                                  - pre_panel['mtb'].mean()),
        'delta_tangibility':    (post_panel['tangibility'].mean()
                                  - pre_panel['tangibility'].mean()),
        'delta_cash':           (post_panel['cash'].mean()
                                  - pre_panel['cash'].mean()),
    }
    records.append(rec)

xs = pd.DataFrame(records)

# Bring in FF12 industry
xs = xs.merge(universe[['gvkey', 'ff12_name']], on='gvkey', how='left')

print(f"\nTurnover events with both pre- and post-CEO data: {len(xs)}")
print(f"Distinct firms: {xs['gvkey'].nunique()}")

xs.to_csv('data/processed_strict/turnover_cross_section.csv', index=False)

# ============================================================
# 3) Descriptive stats
# ============================================================
delta_cols = ['delta_overconf', 'delta_book_leverage', 'delta_debt_share',
              'delta_size', 'delta_profitability', 'delta_mtb',
              'delta_tangibility', 'delta_cash']
print("\nDescriptive stats of changes:")
print(xs[delta_cols].describe().round(3).to_string())

# ============================================================
# 4) Cross-sectional OLS regressions
#    ΔY = β · ΔOverconf + γ · ΔX + ind FE + ν
# ============================================================
def run_xs(outcome, controls=True, industry_fe=True):
    sub = xs.dropna(subset=[outcome, 'delta_overconf']).copy()
    if controls:
        sub = sub.dropna(subset=['delta_size', 'delta_profitability',
                                  'delta_mtb', 'delta_tangibility', 'delta_cash'])
    if industry_fe:
        ind_dummies = pd.get_dummies(sub['ff12_name'], prefix='ind',
                                     drop_first=True).astype(float)
        X = pd.concat([sub[['delta_overconf']], ind_dummies], axis=1)
    else:
        X = sub[['delta_overconf']].copy()
    if controls:
        X = pd.concat([X, sub[['delta_size', 'delta_profitability',
                                'delta_mtb', 'delta_tangibility',
                                'delta_cash']]], axis=1)
    X = sm.add_constant(X)
    y = sub[outcome]
    model = sm.OLS(y, X).fit(cov_type='HC1')
    return model, len(sub)

print("\n" + "=" * 70)
print("H3 — Within-firm change in CEO overconfidence -> change in outcome")
print("=" * 70)

for outcome, label in [('delta_book_leverage', 'Δ book leverage'),
                        ('delta_debt_share',    'Δ debt share (gross)')]:
    print(f"\n--- {label} ---")
    for spec, ctrl, ind in [('(1) Simple', False, False),
                             ('(2) +Industry FE', False, True),
                             ('(3) +Industry FE +Δcontrols', True, True)]:
        m, n = run_xs(outcome, controls=ctrl, industry_fe=ind)
        b = m.params['delta_overconf']
        se = m.bse['delta_overconf']
        p = m.pvalues['delta_overconf']
        stars = ('***' if p < 0.01 else '**' if p < 0.05
                 else '*' if p < 0.10 else '')
        # 95% CI
        lo, hi = m.conf_int().loc['delta_overconf']
        print(f"  {spec:<32} β = {b:+.4f}{stars:<3}  SE = {se:.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  N = {n}")

# ============================================================
# 5) Save the primary spec for the thesis
# ============================================================
m_lev, n_lev = run_xs('delta_book_leverage',  controls=True, industry_fe=True)
m_ds,  n_ds  = run_xs('delta_debt_share',     controls=True, industry_fe=True)

with open('output/strict/h3_regression_results.txt', 'w', encoding='utf-8') as f:
    f.write("H3 - Within-firm CEO turnover (delta overconfidence -> delta outcome)\n")
    f.write("Industry FE + delta controls, HC1 robust SEs\n")
    f.write("=" * 70 + "\n\n")
    f.write("--- delta book leverage ---\n")
    f.write(str(m_lev.summary()))
    f.write("\n\n--- delta debt share (gross) ---\n")
    f.write(str(m_ds.summary()))
print("\nSaved: output/strict/h3_regression_results.txt")