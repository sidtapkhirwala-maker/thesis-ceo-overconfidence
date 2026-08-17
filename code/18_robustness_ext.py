import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

panel = pd.read_csv('data/processed_extended/regression_panel.csv',
                    dtype={'gvkey': str})
xs = pd.read_csv('data/processed_extended/turnover_cross_section.csv',
                 dtype={'gvkey': str})
controls = ['size', 'profitability', 'mtb', 'tangibility', 'cash', 'age']


# ============================================================
# Helpers
# ============================================================
def fmt_coef(beta, se, p, n, label):
    stars = ('***' if p < 0.01 else '**' if p < 0.05
             else '*' if p < 0.10 else '')
    return f"  {label:<55} beta={beta:+.4f}{stars:<3}  SE={se:.4f}  p={p:.3f}  N={n}"


def panel_fe(df, y, x):
    """Firm + year FE panel regression with firm-clustered SEs."""
    sub = df.dropna(subset=[y, x, 'fin_constrained'] + controls).copy()
    if sub.empty or sub['gvkey'].nunique() < 5 or sub['fyear'].nunique() < 2:
        return None, 0
    sub = sub.set_index(['gvkey', 'fyear'])
    formula = (f"{y} ~ {x} + " + " + ".join(controls)
               + " + fin_constrained + EntityEffects + TimeEffects")
    try:
        m = PanelOLS.from_formula(formula, data=sub,
                                  drop_absorbed=True, check_rank=False
                                  ).fit(cov_type='clustered',
                                        cluster_entity=True)
        return m, len(sub)
    except Exception as e:
        print(f"    [!] Panel FE failed: {e}")
        return None, len(sub)


def xs_ols(df, y, x='delta_overconf'):
    """Cross-section OLS for H3 with industry FE and HC1 SEs."""
    sub = df.dropna(subset=[y, x,
                            'delta_size', 'delta_profitability',
                            'delta_mtb', 'delta_tangibility',
                            'delta_cash']).copy()
    if sub.empty or len(sub) < 8:
        return None, len(sub)
    ind_dummies = pd.get_dummies(sub['ff12_name'], prefix='ind',
                                 drop_first=True).astype(float)
    X = pd.concat([sub[[x, 'delta_size', 'delta_profitability',
                        'delta_mtb', 'delta_tangibility', 'delta_cash']],
                   ind_dummies], axis=1)
    X = sm.add_constant(X)
    m = sm.OLS(sub[y], X).fit(cov_type='HC1')
    return m, len(sub)


# ============================================================
# ROBUSTNESS 1 — PCA-aggregated overconfidence index
# ============================================================
print("#" * 75)
print("ROBUSTNESS 1: PCA-aggregated overconfidence index")
print("#" * 75)

for y in ['book_leverage_t1', 'debt_share_t1']:
    m, n = panel_fe(panel, y, 'overconfidence_pca')
    if m is not None:
        b = m.params['overconfidence_pca']
        se = m.std_errors['overconfidence_pca']
        p = m.pvalues['overconfidence_pca']
        print(fmt_coef(b, se, p, n, f"{y} (panel FE, PCA index)"))

# Build H3 delta using PCA index
ov = pd.read_csv('data/processed_extended/overconfidence_index.csv',
                 dtype={'gvkey': str})
ceo_panel = pd.read_csv('data/processed_extended/execucomp_ceo_panel.csv',
                        dtype={'gvkey': str}).rename(columns={'year': 'fyear'})
ov = ov.merge(
    ceo_panel[['gvkey', 'fyear', 'execid']].drop_duplicates(
        subset=['gvkey', 'fyear'], keep='first'),
    on=['gvkey', 'fyear'], how='left'
)
turnovers = pd.read_csv('data/processed_extended/turnover_events.csv',
                        dtype={'gvkey': str})

xs_pca_records = []
for _, ev in turnovers.iterrows():
    gv = ev['gvkey']
    new_id, old_id = ev['execid'], ev['prev_execid']
    firm_ov = ov[ov['gvkey'] == gv]
    pre = firm_ov[firm_ov['execid'] == old_id]['overconfidence_pca']
    post = firm_ov[firm_ov['execid'] == new_id]['overconfidence_pca']
    if len(pre) == 0 or len(post) == 0:
        continue
    xs_pca_records.append({'gvkey': gv,
                           'delta_overconf_pca': post.mean() - pre.mean()})

xs_pca = pd.DataFrame(xs_pca_records)
xs_with_pca = xs.merge(xs_pca, on='gvkey', how='left')

for y in ['delta_book_leverage', 'delta_debt_share']:
    m, n = xs_ols(xs_with_pca, y, x='delta_overconf_pca')
    if m is not None:
        b = m.params['delta_overconf_pca']
        se = m.bse['delta_overconf_pca']
        p = m.pvalues['delta_overconf_pca']
        print(fmt_coef(b, se, p, n, f"{y} (H3, PCA index)"))


# ============================================================
# ROBUSTNESS 2 — Three-way COVID split (Pre / COVID / Post)
# Pre-COVID:   fiscal 2018, 2019
# COVID:       fiscal 2020, 2021
# Post-COVID:  fiscal 2022, 2023, 2024
# This is the central new analysis for the extended sample.
# ============================================================
print("\n" + "#" * 75)
print("ROBUSTNESS 2: Three-way COVID split (Pre / COVID / Post-COVID)")
print("#" * 75)

splits = [
    ('PRE-COVID',  [2018, 2019]),
    ('COVID',      [2020, 2021]),
    ('POST-COVID', [2022, 2023, 2024]),
]

for label, years in splits:
    print(f"\n--- Panel {label} (fyear in {years}) ---")
    sub_panel = panel[panel['fyear'].isin(years)].copy()
    print(f"Sample size: {len(sub_panel)} firm-years, "
          f"{sub_panel['gvkey'].nunique()} firms")
    for y in ['book_leverage_t1', 'debt_share_t1']:
        m, n = panel_fe(sub_panel, y, 'overconfidence')
        if m is not None:
            b = m.params['overconfidence']
            se = m.std_errors['overconfidence']
            p = m.pvalues['overconfidence']
            print(fmt_coef(b, se, p, n, f"{y} ({label})"))


# H3 turnover events split three ways
print("\n--- H3 turnover events by period ---")
for label, years in splits:
    sub_xs = xs[xs['turnover_year'].isin(years)].copy()
    print(f"\n  {label}: {len(sub_xs)} turnover events (years {years})")
    for y in ['delta_book_leverage', 'delta_debt_share']:
        m, n = xs_ols(sub_xs, y)
        if m is not None:
            b = m.params['delta_overconf']
            se = m.bse['delta_overconf']
            p = m.pvalues['delta_overconf']
            print(fmt_coef(b, se, p, n, f"{y} ({label})"))
        else:
            print(f"    [!] {label} {y}: too few events for OLS (N={n})")


# ============================================================
# ROBUSTNESS 3 — Strict-net debt share (sensitivity, expected small N)
# ============================================================
print("\n" + "#" * 75)
print("ROBUSTNESS 3: Strict-net debt share (proposal's original definition)")
print("#" * 75)

if 'debt_share_strict_t1' in panel.columns:
    sub_strict = panel.dropna(subset=['debt_share_strict_t1']).copy()
    print(f"Strict-net sample: {len(sub_strict)} firm-years, "
          f"{sub_strict['gvkey'].nunique()} firms")
    m, n = panel_fe(panel, 'debt_share_strict_t1', 'overconfidence')
    if m is not None:
        b = m.params['overconfidence']
        se = m.std_errors['overconfidence']
        p = m.pvalues['overconfidence']
        print(fmt_coef(b, se, p, n,
                       "debt_share_strict_t1 (panel FE, baseline index)"))
else:
    print("  [!] debt_share_strict_t1 not present in regression panel; skipping.")


# ============================================================
# Save summary
# ============================================================
import os
os.makedirs('output/extended', exist_ok=True)
with open('output/extended/robustness_results.txt', 'w', encoding='utf-8') as f:
    f.write("Robustness battery: PCA index + three-way COVID split + strict-net.\n")
    f.write("Full numerical output is captured by stdout redirection.\n")
print("\nSaved: output/extended/robustness_results.txt")