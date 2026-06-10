import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

panel = pd.read_csv('data/processed/regression_panel.csv',
                    dtype={'gvkey': str})
xs = pd.read_csv('data/processed/turnover_cross_section.csv',
                 dtype={'gvkey': str})
controls = ['size', 'profitability', 'mtb', 'tangibility', 'cash', 'age']


def fmt(name, b, se, p, n):
    stars = ('***' if p < 0.01 else '**' if p < 0.05
             else '*' if p < 0.10 else '')
    return f"  {name:<45} beta={b:+.4f}{stars:<3}  SE={se:.4f}  p={p:.3f}  N={n}"


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


print("#" * 75)
print("ROBUSTNESS 3: Two-component composite (net tone + inverse hedging only)")
print("Drops strong modality given its negative correlation with the other two")
print("#" * 75)

print("\n--- Panel H1 and H2 ---")
for y in ['book_leverage_t1', 'debt_share_t1']:
    m, n = panel_fe(panel, y, 'overconfidence_2comp')
    if m is not None:
        b = m.params['overconfidence_2comp']
        se = m.std_errors['overconfidence_2comp']
        p = m.pvalues['overconfidence_2comp']
        print(fmt(f"{y} (firm+year FE, 2-comp index)", b, se, p, n))

# H3: rebuild Δ-overconfidence using the two-component index
ov = pd.read_csv('data/processed/overconfidence_index.csv',
                 dtype={'gvkey': str})
ceo_panel = pd.read_csv('data/processed/execucomp_ceo_panel.csv',
                        dtype={'gvkey': str}).rename(columns={'year': 'fyear'})
ov = ov.merge(
    ceo_panel[['gvkey', 'fyear', 'execid']]
        .drop_duplicates(subset=['gvkey', 'fyear'], keep='first'),
    on=['gvkey', 'fyear'], how='left'
)
turnovers = pd.read_csv('data/processed/turnover_events.csv',
                        dtype={'gvkey': str})

xs_records = []
for _, ev in turnovers.iterrows():
    gv = ev['gvkey']
    new_id, old_id = ev['execid'], ev['prev_execid']
    firm_ov = ov[ov['gvkey'] == gv]
    pre  = firm_ov[firm_ov['execid'] == old_id]['overconfidence_2comp']
    post = firm_ov[firm_ov['execid'] == new_id]['overconfidence_2comp']
    if len(pre) == 0 or len(post) == 0:
        continue
    xs_records.append({'gvkey': gv,
                        'delta_overconf_2comp': post.mean() - pre.mean()})
xs_2comp = pd.DataFrame(xs_records)
xs_with_2comp = xs.merge(xs_2comp, on='gvkey', how='left')

print("\n--- H3 cross-section (Δ outcome on Δ overconfidence, 2-comp index) ---")
for y in ['delta_book_leverage', 'delta_debt_share']:
    sub = xs_with_2comp.dropna(subset=[y, 'delta_overconf_2comp',
                                       'delta_size', 'delta_profitability',
                                       'delta_mtb', 'delta_tangibility',
                                       'delta_cash']).copy()
    ind_dummies = pd.get_dummies(sub['ff12_name'], prefix='ind',
                                  drop_first=True).astype(float)
    X = pd.concat([sub[['delta_overconf_2comp', 'delta_size', 'delta_profitability',
                         'delta_mtb', 'delta_tangibility', 'delta_cash']],
                   ind_dummies], axis=1)
    X = sm.add_constant(X)
    m = sm.OLS(sub[y], X).fit(cov_type='HC1')
    b = m.params['delta_overconf_2comp']
    se = m.bse['delta_overconf_2comp']
    p = m.pvalues['delta_overconf_2comp']
    print(fmt(f"{y} (industry FE + delta controls, 2-comp)", b, se, p, len(sub)))

# Save summary
with open('output/two_component_robustness.txt', 'w', encoding='utf-8') as f:
    f.write("Robustness 3: Two-component composite (net tone + inverse hedging)\n")
    f.write("See terminal output for full results.\n")
print("\nSaved: output/two_component_robustness.txt")