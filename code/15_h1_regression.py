import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

# ============================================================
# 1) Load the regression panel
# ============================================================
panel = pd.read_csv('data/processed/regression_panel.csv',
                    dtype={'gvkey': str})
print(f"Panel rows: {len(panel)}")

# linearmodels needs gvkey + fyear as MultiIndex; fyear must be numeric
panel = panel.dropna(subset=['book_leverage_t1', 'overconfidence',
                              'size', 'profitability', 'mtb',
                              'tangibility', 'cash', 'age']).copy()
panel = panel.set_index(['gvkey', 'fyear'])
print(f"Panel after dropping NA on H1 vars: {len(panel)}")

# ============================================================
# 2) Specification ladder for H1
#    (1) Overconfidence only, no FE
#    (2) + controls, no FE
#    (3) + year FE
#    (4) + firm + year FE (the proposal's primary spec)
# ============================================================
controls = ['size', 'profitability', 'mtb', 'tangibility', 'cash', 'age']
y = 'book_leverage_t1'

def fit(formula, **kw):
    return PanelOLS.from_formula(formula, data=panel,
                                 drop_absorbed=True, check_rank=False
                                 ).fit(cov_type='clustered',
                                       cluster_entity=True, **kw)

print("\n" + "=" * 70)
print("H1 — Book leverage(t+1) on CEO overconfidence(t)")
print("=" * 70)

m1 = fit(f"{y} ~ 1 + overconfidence")
m2 = fit(f"{y} ~ 1 + overconfidence + " + " + ".join(controls))
m3 = fit(f"{y} ~ 1 + overconfidence + " + " + ".join(controls) + " + TimeEffects")
m4 = fit(f"{y} ~ overconfidence + " + " + ".join(controls)
         + " + EntityEffects + TimeEffects")

# ============================================================
# 3) Pretty-print the results side-by-side
# ============================================================
def row(name, models, attr):
    cells = []
    for m in models:
        if name in m.params.index:
            beta = m.params[name]
            se   = m.std_errors[name]
            pval = m.pvalues[name]
            stars = ('***' if pval < 0.01 else '**' if pval < 0.05
                     else '*' if pval < 0.10 else '')
            if attr == 'beta':
                cells.append(f"{beta:>8.4f}{stars:<3}")
            else:
                cells.append(f" ({se:>6.4f})  ")
        else:
            cells.append(" " * 13)
    return cells

models = [m1, m2, m3, m4]
labels = ['(1) Simple', '(2) +Controls', '(3) +Year FE', '(4) +Firm+Year FE']

vars_to_show = ['overconfidence'] + controls
print(f"\n{'Variable':<18} " + " ".join(f"{l:<13}" for l in labels))
print("-" * 75)
for v in vars_to_show:
    bcells = row(v, models, 'beta')
    secells = row(v, models, 'se')
    print(f"{v:<18} " + " ".join(bcells))
    print(f"{'':<18} " + " ".join(secells))

print("-" * 75)
print(f"{'N':<18} " + " ".join(f"{int(m.nobs):<13}" for m in models))
print(f"{'R² (within)':<18} " + " ".join(f"{m.rsquared_within:<13.4f}"
                                          if hasattr(m, 'rsquared_within')
                                          and not np.isnan(m.rsquared_within)
                                          else f"{m.rsquared:<13.4f}"
                                          for m in models))
print(f"{'Firm FE':<18} " + " ".join(f"{'No':<13}" for _ in range(3))
      + f"{'Yes':<13}")
print(f"{'Year FE':<18} " + f"{'No':<13}" * 2
      + f"{'Yes':<13}" * 2)
print("\nStars: *** p<0.01, ** p<0.05, * p<0.10")
print("Standard errors clustered at the firm level.")

# ============================================================
# 4) Save the primary spec for the thesis
# ============================================================
with open('output/h1_regression_results.txt', 'w') as f:
    f.write("H1 — Book leverage(t+1) on CEO overconfidence(t)\n")
    f.write("Primary specification: firm + year FE, firm-clustered SEs\n")
    f.write("=" * 70 + "\n\n")
    f.write(str(m4))

print("\nSaved primary spec to: output/h1_regression_results.txt")