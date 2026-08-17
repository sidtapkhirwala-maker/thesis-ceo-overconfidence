import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

# ============================================================
# Load panel
# ============================================================
panel = pd.read_csv('data/processed_strict/regression_panel.csv',
                    dtype={'gvkey': str})

controls = ['size', 'profitability', 'mtb', 'tangibility', 'cash', 'age']


def run_h2(outcome_col, label):
    sub = panel.dropna(subset=[outcome_col, 'overconfidence',
                               'fin_constrained'] + controls).copy()
    sub = sub.set_index(['gvkey', 'fyear'])

    n = len(sub)
    n_firms = sub.index.get_level_values(0).nunique()
    n_years = sub.index.get_level_values(1).nunique()

    print("\n" + "=" * 70)
    print(f"H2 — {label}: {outcome_col}(t+1) on overconfidence(t)")
    print(f"N = {n} firm-years; distinct firms = {n_firms}; years = {n_years}")
    print("=" * 70)

    def fit(formula):
        return PanelOLS.from_formula(formula, data=sub,
                                     drop_absorbed=True, check_rank=False
                                     ).fit(cov_type='clustered',
                                           cluster_entity=True)

    base = " + ".join(controls + ['fin_constrained'])
    m1 = fit(f"{outcome_col} ~ 1 + overconfidence")
    m2 = fit(f"{outcome_col} ~ 1 + overconfidence + {base}")
    m3 = fit(f"{outcome_col} ~ 1 + overconfidence + {base} + TimeEffects")

    # Column 4 (firm + year FE) requires enough degrees of freedom.
    # If sample is too small, fall back to a lean spec.
    df_used = n_firms + n_years + len(controls) + 2  # 2 = overconf + fin_constrained
    if n - df_used > 5:
        m4 = fit(f"{outcome_col} ~ overconfidence + {base}"
                 " + EntityEffects + TimeEffects")
        col4_label = '(4) +Firm+Year FE'
    else:
        print(f"\n[!] Sample too small (N={n}, firms={n_firms}, years={n_years}) "
              f"for firm+year FE. Falling back to year FE only.")
        m4 = fit(f"{outcome_col} ~ 1 + overconfidence + {base} + TimeEffects")
        col4_label = '(4) Year FE only (lean)'

    models = [m1, m2, m3, m4]
    labels = ['(1) Simple', '(2) +Controls', '(3) +Year FE', col4_label]

    vars_to_show = ['overconfidence'] + controls + ['fin_constrained']
    print(f"\n{'Variable':<18} " + " ".join(f"{l:<20}" for l in labels))
    print("-" * 95)
    for v in vars_to_show:
        bcells, secells = [], []
        for m in models:
            if v in m.params.index:
                beta = m.params[v]
                se   = m.std_errors[v]
                pval = m.pvalues[v]
                stars = ('***' if pval < 0.01 else '**' if pval < 0.05
                         else '*' if pval < 0.10 else '')
                bcells.append(f"{beta:>10.4f}{stars:<3}      ")
                secells.append(f"  ({se:>8.4f})        ")
            else:
                bcells.append(" " * 20)
                secells.append(" " * 20)
        print(f"{v:<18} " + " ".join(bcells))
        print(f"{'':<18} " + " ".join(secells))

    print("-" * 95)
    print(f"{'N':<18} " + " ".join(f"{int(m.nobs):<20}" for m in models))
    print("\nStars: *** p<0.01, ** p<0.05, * p<0.10")
    print("Standard errors clustered at the firm level.")
    return m4


# ============================================================
# H2 BASELINE — Gross issuance debt share
# ============================================================
print("\n" + "#" * 70)
print("# H2 BASELINE: GROSS issuance debt share")
print("# debt_share = dltis / (dltis + sstk)")
print("#" * 70)
m_baseline = run_h2('debt_share_t1', 'BASELINE (gross issuance)')

# ============================================================
# H2 ROBUSTNESS — Strict net issuance debt share
# (proposal's literal definition, but very small sample)
# ============================================================
print("\n" + "#" * 70)
print("# H2 ROBUSTNESS: STRICT NET issuance debt share (proposal's literal definition)")
print("# debt_share = (dltis-dltr) / [(dltis-dltr) + (sstk-prstkc)], both non-neg")
print("#" * 70)
m_strict = run_h2('debt_share_strict_t1', 'STRICT NET robustness')

# ============================================================
# Save primary specs
# ============================================================
with open('output/strict/h2_regression_results.txt', 'w') as f:
    f.write("H2 BASELINE (gross issuance debt share)\n")
    f.write("=" * 70 + "\n\n")
    f.write(str(m_baseline))
    f.write("\n\n\n")
    f.write("H2 ROBUSTNESS (strict net issuance debt share)\n")
    f.write("=" * 70 + "\n\n")
    f.write(str(m_strict))

print("\nSaved: output/strict/h2_regression_results.txt")