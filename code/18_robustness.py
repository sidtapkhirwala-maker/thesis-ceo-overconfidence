import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

panel = pd.read_csv('data/processed/regression_panel.csv',
                    dtype={'gvkey': str})
xs = pd.read_csv('data/processed/turnover_cross_section.csv',
                 dtype={'gvkey': str})
controls = ['size', 'profitability', 'mtb', 'tangibility', 'cash', 'age']

# Helper for printing a single coefficient row
def fmt_coef(beta, se, p, n, label):
    stars = ('***' if p < 0.01 else '**' if p < 0.05
             else '*' if p < 0.10 else '')
    return f"  {label:<45} beta={beta:+.4f}{stars:<3}  SE={se:.4f}  N={n}"

# Panel-FE workhorse for H1/H2
def panel_fe(df, y, x):
    sub = df.dropna(subset=[y, x, 'fin_constrained'] + controls).copy()
    if sub.empty:
        return None, 0
    sub = sub.set_index(['gvkey', 'fyear'])
    formula = (f"{y} ~ {x} + " + " + ".join(controls)
               + " + fin_constrained + EntityEffects + TimeEffects")
    m = PanelOLS.from_formula(formula, data=sub,
                              drop_absorbed=True, check_rank=False
                              ).fit(cov_type='clustered', cluster_entity=True)
    return m, len(sub)

# Cross-section OLS workhorse for H3
def xs_ols(df, y):
    sub = df.dropna(subset=[y, 'delta_overconf',
                            'delta_size', 'delta_profitability',
                            'delta_mtb', 'delta_tangibility',
                            'delta_cash']).copy()
    if sub.empty:
        return None, 0
    ind_dummies = pd.get_dummies(sub['ff12_name'], prefix='ind',
                                  drop_first=True).astype(float)
    X = pd.concat([sub[['delta_overconf', 'delta_size', 'delta_profitability',
                         'delta_mtb', 'delta_tangibility', 'delta_cash']],
                   ind_dummies], axis=1)
    X = sm.add_constant(X)
    m = sm.OLS(sub[y], X).fit(cov_type='HC1')
    return m, len(sub)

print("#" * 70)
print("ROBUSTNESS 1: PCA-aggregated overconfidence index")
print("#" * 70)

for y in ['book_leverage_t1', 'debt_share_t1']:
    m, n = panel_fe(panel, y, 'overconfidence_pca')
    if m is not None:
        b = m.params['overconfidence_pca']
        se = m.std_errors['overconfidence_pca']
        p = m.pvalues['overconfidence_pca']
        print(fmt_coef(b, se, p, n, f"{y} (panel FE, PCA index)"))

# For H3 we re-derive a PCA delta. Build delta from per-call PCA component:
ov = pd.read_csv('data/processed/overconfidence_index.csv',
                 dtype={'gvkey': str})
ceo_panel = pd.read_csv('data/processed/execucomp_ceo_panel.csv',
                        dtype={'gvkey': str}).rename(columns={'year': 'fyear'})
ov = ov.merge(ceo_panel[['gvkey', 'fyear', 'execid']].drop_duplicates(
    subset=['gvkey', 'fyear'], keep='first'),
    on=['gvkey', 'fyear'], how='left')
turnovers = pd.read_csv('data/processed/turnover_events.csv',
                        dtype={'gvkey': str})

xs_pca_records = []
for _, ev in turnovers.iterrows():
    gv = ev['gvkey']
    new_id = ev['execid']
    old_id = ev['prev_execid']
    firm_ov = ov[ov['gvkey'] == gv]
    pre  = firm_ov[firm_ov['execid'] == old_id]['overconfidence_pca']
    post = firm_ov[firm_ov['execid'] == new_id]['overconfidence_pca']
    if len(pre) == 0 or len(post) == 0:
        continue
    xs_pca_records.append({'gvkey': gv,
                            'delta_overconf_pca': post.mean() - pre.mean()})
xs_pca = pd.DataFrame(xs_pca_records)
xs_with_pca = xs.merge(xs_pca, on='gvkey', how='left')

for y in ['delta_book_leverage', 'delta_debt_share']:
    sub = xs_with_pca.dropna(subset=[y, 'delta_overconf_pca',
                                      'delta_size', 'delta_profitability',
                                      'delta_mtb', 'delta_tangibility',
                                      'delta_cash']).copy()
    ind_dummies = pd.get_dummies(sub['ff12_name'], prefix='ind',
                                  drop_first=True).astype(float)
    X = pd.concat([sub[['delta_overconf_pca', 'delta_size', 'delta_profitability',
                         'delta_mtb', 'delta_tangibility', 'delta_cash']],
                   ind_dummies], axis=1)
    X = sm.add_constant(X)
    m = sm.OLS(sub[y], X).fit(cov_type='HC1')
    b = m.params['delta_overconf_pca']
    se = m.bse['delta_overconf_pca']
    p = m.pvalues['delta_overconf_pca']
    print(fmt_coef(b, se, p, len(sub), f"{y} (H3, PCA index)"))

print("\n" + "#" * 70)
print("ROBUSTNESS 2: Pre/Post-COVID split (drop fyear 2020 and 2021)")
print("#" * 70)

print("\n--- Panel restricted to fyear in [2018, 2019, 2022] ---")
panel_noCovid = panel[panel['fyear'].isin([2018, 2019, 2022])].copy()
print(f"Sample size: {len(panel_noCovid)} firm-years")
for y in ['book_leverage_t1', 'debt_share_t1']:
    m, n = panel_fe(panel_noCovid, y, 'overconfidence')
    if m is not None:
        b = m.params['overconfidence']
        se = m.std_errors['overconfidence']
        p = m.pvalues['overconfidence']
        print(fmt_coef(b, se, p, n, f"{y} (no COVID)"))

print("\n--- Panel COVID-only fyear in [2020, 2021] ---")
panel_Covid = panel[panel['fyear'].isin([2020, 2021])].copy()
print(f"Sample size: {len(panel_Covid)} firm-years")
for y in ['book_leverage_t1', 'debt_share_t1']:
    m, n = panel_fe(panel_Covid, y, 'overconfidence')
    if m is not None:
        b = m.params['overconfidence']
        se = m.std_errors['overconfidence']
        p = m.pvalues['overconfidence']
        print(fmt_coef(b, se, p, n, f"{y} (COVID only)"))

print("\n--- H3 restricted to turnovers in [2018, 2019, 2022] ---")
xs_noCovid = xs[xs['turnover_year'].isin([2018, 2019, 2022])].copy()
print(f"Turnover events: {len(xs_noCovid)}")
for y in ['delta_book_leverage', 'delta_debt_share']:
    m, n = xs_ols(xs_noCovid, y)
    if m is not None:
        b = m.params['delta_overconf']
        se = m.bse['delta_overconf']
        p = m.pvalues['delta_overconf']
        print(fmt_coef(b, se, p, n, f"{y} (no COVID)"))

# Save summary
with open('output/robustness_results.txt', 'w', encoding='utf-8') as f:
    f.write("Robustness battery: PCA index + pre/post-COVID split\n")
    f.write("See terminal output for full results.\n")
print("\nSaved: output/robustness_results.txt")