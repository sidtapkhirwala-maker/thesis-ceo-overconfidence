import wrds
import pandas as pd
import numpy as np
from sqlalchemy import text

db = wrds.Connection(wrds_username='sidt28')

def safe_sql(query):
    with db.engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
        cols = list(result.keys())
    return pd.DataFrame(rows, columns=cols)

# ============================================================
# 1) Load the surviving firm-years from the corpus filter
# ============================================================
fy_keep = pd.read_csv('data/processed_strict/final_firm_years.csv',
                      dtype={'gvkey': str})
fy_keep = fy_keep.rename(columns={'call_year': 'fyear'})
print(f"Surviving firm-years (from corpus filter): {len(fy_keep)}")

# ============================================================
# 2) Load the overconfidence index
# ============================================================
ov = pd.read_csv('data/processed_strict/overconfidence_index.csv',
                 dtype={'gvkey': str})
print(f"Firm-years with overconfidence index: {len(ov)}")

# ============================================================
# 3) Load the Compustat panel and clean
# ============================================================
funda = pd.read_csv('data/processed_strict/compustat_panel.csv',
                    dtype={'gvkey': str})

# Pick one row per gvkey-fyear (some firms have multiple due to FY-end changes)
funda = funda.sort_values(['gvkey', 'fyear', 'datadate']).drop_duplicates(
    subset=['gvkey', 'fyear'], keep='last'
)

# ============================================================
# 4) Construct outcomes and controls
# ============================================================
# Book leverage (primary H1 outcome)
funda['book_leverage'] = (funda['dltt'] + funda['dlc']) / funda['at']

# Debt share: TWO operationalisations
# ---- Baseline (H2): GROSS issuances ----
# Modern S&P 500 firms run continuous share repurchases, making
# `sstk - prstkc` almost always negative. Gross issuances cleanly answer
# the MTY question: when raising fresh outside capital, debt or equity?
gross_external = funda['dltis'] + funda['sstk']
funda['debt_share'] = np.where(
    gross_external > 0,
    funda['dltis'] / gross_external,
    np.nan
)

# ---- Robustness: STRICT NET (the proposal's literal definition) ----
funda['net_debt']     = funda['dltis'] - funda['dltr']
funda['net_equity']   = funda['sstk']  - funda['prstkc']
funda['net_external'] = funda['net_debt'] + funda['net_equity']
both_nonneg = (funda['net_debt'] >= 0) & (funda['net_equity'] >= 0)
funda['debt_share_strict'] = np.where(
    both_nonneg & (funda['net_external'] > 0),
    funda['net_debt'] / funda['net_external'],
    np.nan
)

# Controls
funda['size']          = np.log(funda['at'])
funda['profitability'] = funda['oibdp'] / funda['at']
funda['mtb']           = (funda['prcc_f'] * funda['csho']
                          + funda['dltt'] + funda['dlc']) / funda['at']
funda['tangibility']   = funda['ppent'] / funda['at']
funda['cash']          = funda['che']   / funda['at']

# ============================================================
# 5) TRUE firm age: pull each firm's first-ever fyear from Compustat
# ============================================================
gvkeys = tuple(funda['gvkey'].unique().tolist())
print("\nFetching each firm's first Compustat appearance year...")
first_years = safe_sql(f"""
    SELECT gvkey, MIN(fyear) AS first_fyear
    FROM comp.funda
    WHERE gvkey IN {gvkeys}
      AND indfmt='INDL' AND datafmt='STD'
      AND popsrc='D' AND consol='C'
      AND fyear IS NOT NULL
    GROUP BY gvkey
""")
first_years['gvkey']       = first_years['gvkey'].astype(str)
first_years['first_fyear'] = first_years['first_fyear'].astype(int)

funda = funda.merge(first_years, on='gvkey', how='left')
funda['age'] = funda['fyear'] - funda['first_fyear']

print("Firm age sanity check (years since first Compustat appearance):")
print(funda.groupby('gvkey')['age'].max().describe().round(1).to_string())

# ============================================================
# 6) SA financial-constraint index (Hadlock-Pierce 2010)
#    SA = -0.737*size_w + 0.043*size_w^2 - 0.040*age_c
#    size winsorised at $4.5B (log = 8.41 when 'at' is in $millions)
#    age capped at 37 years
# ============================================================
SA_SIZE_CAP_LOG = np.log(4500)
SA_AGE_CAP      = 37

funda['size_sa']  = funda['size'].clip(upper=SA_SIZE_CAP_LOG)
funda['age_sa']   = funda['age'].clip(upper=SA_AGE_CAP)
funda['sa_index'] = (-0.737 * funda['size_sa']
                     + 0.043 * funda['size_sa']**2
                     - 0.040 * funda['age_sa'])

sa_top_thr = funda['sa_index'].quantile(2/3)
funda['fin_constrained'] = (funda['sa_index'] >= sa_top_thr).astype(int)

# ============================================================
# 7) Build the lagged structure: overconfidence(t) -> outcome(t+1)
# ============================================================
funda_panel = funda[funda['fyear'].between(2017, 2025)].copy()

# Year-t controls
controls_t = funda_panel[['gvkey', 'fyear', 'size', 'profitability',
                          'mtb', 'tangibility', 'cash', 'age',
                          'fin_constrained', 'sa_index']].copy()

# Year-t+1 outcomes (rename fyear to t for the merge)
outcomes_t1 = funda_panel[['gvkey', 'fyear', 'book_leverage',
                            'debt_share', 'debt_share_strict',
                            'net_external']].copy()
outcomes_t1 = outcomes_t1.rename(columns={
    'fyear': 'fyear_outcome',
    'book_leverage':     'book_leverage_t1',
    'debt_share':        'debt_share_t1',
    'debt_share_strict': 'debt_share_strict_t1',
    'net_external':      'net_external_t1',
})
outcomes_t1['fyear'] = outcomes_t1['fyear_outcome'] - 1

# ============================================================
# 8) Merge: overconfidence(t) + controls(t) + outcomes(t+1)
# ============================================================
panel = (
    ov[['gvkey', 'fyear', 'overconfidence', 'overconfidence_pca', 'overconfidence_2comp',
        'z_net_tone', 'z_strong_modality', 'z_inverse_hedging']]
    .merge(controls_t, on=['gvkey', 'fyear'], how='inner')
    .merge(outcomes_t1[['gvkey', 'fyear',
                        'book_leverage_t1',
                        'debt_share_t1', 'debt_share_strict_t1',
                        'net_external_t1']],
           on=['gvkey', 'fyear'], how='left')
)

# Restrict to firm-years that survived the corpus filter
panel = panel.merge(fy_keep[['gvkey', 'fyear']],
                    on=['gvkey', 'fyear'], how='inner')

print(f"\nFinal regression panel rows: {len(panel)}")
print(f"Distinct firms: {panel['gvkey'].nunique()}")
print(f"Year coverage:")
print(panel.groupby('fyear').size().to_string())

# ============================================================
# 9) Winsorise continuous controls + outcomes at 1% / 99%
# ============================================================
for col in ['profitability', 'mtb', 'tangibility', 'cash',
            'book_leverage_t1', 'debt_share_t1', 'debt_share_strict_t1']:
    if col in panel.columns:
        lo, hi = panel[col].quantile([0.01, 0.99])
        panel[col] = panel[col].clip(lo, hi)

# ============================================================
# 10) Diagnostics
# ============================================================
print(f"\nDescriptive stats of the panel:")
cols_to_show = ['overconfidence', 'book_leverage_t1',
                'debt_share_t1', 'debt_share_strict_t1',
                'size', 'profitability', 'mtb', 'tangibility', 'cash',
                'age', 'fin_constrained']
print(panel[cols_to_show].describe().round(3).to_string())

print(f"\nH1 sample size (book_leverage_t1 non-missing): "
      f"{panel['book_leverage_t1'].notna().sum()}")
print(f"H2 sample size (debt_share_t1 - GROSS baseline): "
      f"{panel['debt_share_t1'].notna().sum()}")
print(f"H2 sample size (debt_share_strict_t1 - NET strict robustness): "
      f"{panel['debt_share_strict_t1'].notna().sum()}")

# ============================================================
# 11) Save the merged panel
# ============================================================
panel.to_csv('data/processed_strict/regression_panel.csv', index=False)
print(f"\nSaved: data/processed_strict/regression_panel.csv")

db.close()