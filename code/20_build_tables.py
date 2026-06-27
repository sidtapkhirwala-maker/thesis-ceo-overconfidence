import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import os

os.makedirs('output/tables', exist_ok=True)

# ============================================================
# Load data
# ============================================================
panel = pd.read_csv('data/processed/regression_panel.csv',
                    dtype={'gvkey': str})
xs = pd.read_csv('data/processed/turnover_cross_section.csv',
                 dtype={'gvkey': str})
controls = ['size', 'profitability', 'mtb', 'tangibility', 'cash', 'age']

# ============================================================
# Helpers
# ============================================================
def stars(p):
    return ('***' if p < 0.01 else '**' if p < 0.05
            else '*' if p < 0.10 else '')

def fmt_coef(beta, se, p):
    return f"{beta:+.4f}{stars(p)}"

def fmt_se(se):
    return f"({se:.4f})"

def fit_panel(df, y, x_vars, entity_fe=False, time_fe=False):
    sub = df.dropna(subset=[y] + x_vars).copy()
    sub = sub.set_index(['gvkey', 'fyear'])
    rhs = ' + '.join(x_vars)
    if entity_fe and time_fe:
        formula = f"{y} ~ {rhs} + EntityEffects + TimeEffects"
    elif time_fe:
        formula = f"{y} ~ 1 + {rhs} + TimeEffects"
    elif entity_fe:
        formula = f"{y} ~ {rhs} + EntityEffects"
    else:
        formula = f"{y} ~ 1 + {rhs}"
    m = PanelOLS.from_formula(formula, data=sub,
                              drop_absorbed=True, check_rank=False
                              ).fit(cov_type='clustered', cluster_entity=True)
    return m

VAR_LABELS = {
    'overconfidence':     'Overconfidence',
    'overconfidence_pca': 'Overconfidence (PCA)',
    'overconfidence_2comp':'Overconfidence (2-comp)',
    'size':               'Size (log assets)',
    'profitability':      'Profitability',
    'mtb':                'Market-to-book',
    'tangibility':        'Tangibility',
    'cash':               'Cash',
    'age':                'Firm age',
    'fin_constrained':    'Financially constrained',
    'delta_overconf':     'Delta Overconfidence',
    'delta_size':         'Delta Size',
    'delta_profitability':'Delta Profitability',
    'delta_mtb':          'Delta M/B',
    'delta_tangibility':  'Delta Tangibility',
    'delta_cash':         'Delta Cash',
}

def build_panel_table(outcome, specs, var_order, title, notes):
    """specs is list of (label, x_vars, entity_fe, time_fe)"""
    models = [(label, fit_panel(panel, outcome, x_vars, ef, tf))
              for label, x_vars, ef, tf in specs]

    rows = []
    rows.append([title] + [''] * len(models))
    rows.append(['Variable'] + [lab for lab, _ in models])
    for v in var_order:
        coef_row = [VAR_LABELS.get(v, v)]
        se_row   = ['']
        for _, m in models:
            if v in m.params.index:
                coef_row.append(fmt_coef(m.params[v], m.std_errors[v],
                                          m.pvalues[v]))
                se_row.append(fmt_se(m.std_errors[v]))
            else:
                coef_row.append('—')
                se_row.append('')
        rows.append(coef_row)
        rows.append(se_row)
    rows.append(['Observations'] + [f"{int(m.nobs)}" for _, m in models])
    rows.append(['R-squared (within)']
                + [f"{m.rsquared_within:.4f}" if not np.isnan(m.rsquared_within)
                    else f"{m.rsquared:.4f}" for _, m in models])
    rows.append(['Firm fixed effects']  + [('Yes' if ef else 'No') for _, _, ef, _ in specs])
    rows.append(['Year fixed effects']  + [('Yes' if tf else 'No') for _, _, _, tf in specs])
    rows.append(['Notes:', notes] + [''] * (len(models) - 1))
    return rows

# ============================================================
# Table 1: H1 — Book leverage
# ============================================================
h1_specs = [
    ('(1)',  ['overconfidence'],              False, False),
    ('(2)',  ['overconfidence'] + controls,   False, False),
    ('(3)',  ['overconfidence'] + controls,   False, True),
    ('(4)',  ['overconfidence'] + controls,   True,  True),
]
h1_table = build_panel_table(
    outcome='book_leverage_t1',
    specs=h1_specs,
    var_order=['overconfidence'] + controls,
    title='Table 1: CEO Linguistic Overconfidence and Book Leverage (H1)',
    notes='Dependent variable: book leverage at t+1. Firm-clustered SEs in parentheses. *** p<0.01, ** p<0.05, * p<0.10.'
)
pd.DataFrame(h1_table).to_csv('output/tables/table_1_h1.csv',
                              index=False, header=False)

# ============================================================
# Table 2: H2 — Debt share (gross)
# ============================================================
h2_specs = [
    ('(1)',  ['overconfidence'],                                     False, False),
    ('(2)',  ['overconfidence'] + controls + ['fin_constrained'],    False, False),
    ('(3)',  ['overconfidence'] + controls + ['fin_constrained'],    False, True),
    ('(4)',  ['overconfidence'] + controls + ['fin_constrained'],    True,  True),
]
h2_table = build_panel_table(
    outcome='debt_share_t1',
    specs=h2_specs,
    var_order=['overconfidence'] + controls + ['fin_constrained'],
    title='Table 2: CEO Linguistic Overconfidence and Debt Share of Gross External Financing (H2)',
    notes='Dependent variable: debt share at t+1 (gross-issuance operationalisation, see Section 5). Firm-clustered SEs in parentheses. *** p<0.01, ** p<0.05, * p<0.10.'
)
pd.DataFrame(h2_table).to_csv('output/tables/table_2_h2.csv',
                              index=False, header=False)

# ============================================================
# Table 3: H3 — CEO turnover cross-section
# ============================================================
def fit_xs(y, controls_flag, indfe_flag):
    sub = xs.dropna(subset=[y, 'delta_overconf']).copy()
    cols = ['delta_overconf']
    if controls_flag:
        sub = sub.dropna(subset=['delta_size', 'delta_profitability',
                                  'delta_mtb', 'delta_tangibility', 'delta_cash'])
        cols += ['delta_size', 'delta_profitability', 'delta_mtb',
                 'delta_tangibility', 'delta_cash']
    X = sub[cols].copy()
    if indfe_flag:
        ind_dummies = pd.get_dummies(sub['ff12_name'], prefix='ind',
                                      drop_first=True).astype(float)
        X = pd.concat([X, ind_dummies], axis=1)
    X = sm.add_constant(X)
    return sm.OLS(sub[y], X).fit(cov_type='HC1'), len(sub)

h3_specs_xs = [
    ('(1)',  False, False),
    ('(2)',  False, True),
    ('(3)',  True,  True),
]
h3_rows = []
h3_rows.append(['Table 3: Within-firm Change in Overconfidence Around CEO Turnover (H3)'] + [''] * 6)
h3_rows.append(['', 'Delta Book Leverage', '', '', 'Delta Debt Share', '', ''])
h3_rows.append(['Variable'] + ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)'])

models_lev = [fit_xs('delta_book_leverage', c, ind) for _, c, ind in h3_specs_xs]
models_ds  = [fit_xs('delta_debt_share',    c, ind) for _, c, ind in h3_specs_xs]
all_models = models_lev + models_ds

xs_var_order = ['delta_overconf', 'delta_size', 'delta_profitability',
                'delta_mtb', 'delta_tangibility', 'delta_cash']

for v in xs_var_order:
    coef_row = [VAR_LABELS.get(v, v)]
    se_row   = ['']
    for m, _ in all_models:
        if v in m.params.index:
            coef_row.append(fmt_coef(m.params[v], m.bse[v], m.pvalues[v]))
            se_row.append(fmt_se(m.bse[v]))
        else:
            coef_row.append('—')
            se_row.append('')
    h3_rows.append(coef_row)
    h3_rows.append(se_row)

h3_rows.append(['Observations'] + [f"{int(m.nobs)}" for m, _ in all_models])
h3_rows.append(['R-squared'] + [f"{m.rsquared:.4f}" for m, _ in all_models])
h3_rows.append(['Industry FE'] + ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes'])
h3_rows.append(['Delta controls'] + ['No', 'No', 'Yes', 'No', 'No', 'Yes'])
h3_rows.append(['Notes:', 'HC1 robust SEs in parentheses. *** p<0.01, ** p<0.05, * p<0.10.']
                + [''] * 5)
pd.DataFrame(h3_rows).to_csv('output/tables/table_3_h3.csv',
                              index=False, header=False)

# ============================================================
# Table 4: Descriptive statistics
# ============================================================
desc_vars = ['overconfidence', 'book_leverage_t1', 'debt_share_t1',
             'size', 'profitability', 'mtb', 'tangibility', 'cash',
             'age', 'fin_constrained']
desc = panel[desc_vars].describe().T[['count', 'mean', 'std',
                                       'min', '25%', '50%', '75%', 'max']]
desc.columns = ['N', 'Mean', 'Std Dev', 'Min', 'P25', 'Median', 'P75', 'Max']
desc.index = [VAR_LABELS.get(v, v) for v in desc.index]
desc = desc.round(3)
desc.to_csv('output/tables/table_4_descriptives.csv')

# ============================================================
# Table 5: Component correlation matrix
# ============================================================
ov = pd.read_csv('data/processed/overconfidence_index.csv',
                  dtype={'gvkey': str})
corr_vars = ['z_net_tone', 'z_strong_modality', 'z_inverse_hedging']
corr_labels = ['Net tone (z)', 'Strong modality (z)', 'Inverse hedging (z)']
corr = ov[corr_vars].corr().round(3)
corr.index = corr_labels
corr.columns = corr_labels
corr.to_csv('output/tables/table_5_correlations.csv')

# ============================================================
# Table 6: Robustness battery summary
# ============================================================
rob = [
    ['Table 6: Robustness Battery — Coefficient on Overconfidence Across Specifications',
     '', '', '', ''],
    ['', 'H1 panel (Δleverage)', 'H2 panel (Δdebt share)',
     'H3 Δ leverage', 'H3 Δ debt share'],
    ['Baseline (3-component, simple-avg)',
     '+0.0133 (NS)', '+0.0694 (NS)', '+0.0150 (NS)', '+0.1850 (NS)'],
    ['PCA-aggregated index',
     '+0.0026 (NS)', '+0.0540 (p≈0.09)*', '+0.0010 (NS)', '−0.0049 (NS)'],
    ['Two-component (no strong-modal)',
     '+0.0067 (NS)', '+0.0813 (p≈0.08)*', '+0.0048 (NS)', '−0.0005 (NS)'],
    ['No-COVID sub-sample (2018, 2019, 2022)',
     '+0.0109 (NS)', '−0.0395 (NS)', '+0.0441 (NS)', '−0.1407 (NS)'],
    ['COVID-only sub-sample (2020, 2021)',
     '+0.0058 (NS)', '+0.1038 (NS)', '—', '—'],
    ['Notes:',
     'NS = not significant. * p<0.10. Each row is the coefficient on the overconfidence index from the same panel specification estimated under the alternative index or sub-sample.',
     '', '', ''],
]
pd.DataFrame(rob).to_csv('output/tables/table_6_robustness.csv',
                          index=False, header=False)

print("\nAll tables saved to output/tables/:")
print("  table_1_h1.csv          — H1 specification ladder")
print("  table_2_h2.csv          — H2 specification ladder")
print("  table_3_h3.csv          — H3 turnover cross-section")
print("  table_4_descriptives.csv — Panel summary statistics")
print("  table_5_correlations.csv — Component correlation matrix")
print("  table_6_robustness.csv  — Robustness battery summary")