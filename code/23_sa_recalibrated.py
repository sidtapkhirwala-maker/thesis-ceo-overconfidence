import pandas as pd
import numpy as np

# ============================================================
# Robustness check: SA financial-constraint index recalibrated
# for an S&P 500-only sample.
#
# The original SA index (code/14_build_regression_panel_ext.py)
# winsorises size at log($4.5B) and age at 37 years, following
# Hadlock & Pierce (2010)'s broader, smaller-firm calibration
# sample. In an S&P 500-only sample most firm-years exceed both
# caps, so a large share of observations collapse onto an
# identical SA score, undermining the top-tercile "constrained"
# classification. This script recomputes SA using caps set at
# the 95th percentile of size/age WITHIN this sample instead.
#
# `regression_panel.csv` already carries the RAW (uncapped)
# 'size' (=log(at)) and 'age' columns alongside the original
# capped 'sa_index' / 'fin_constrained', so no re-pull from
# WRDS is needed here.
# ============================================================

panel = pd.read_csv('data/processed_extended/regression_panel.csv',
                    dtype={'gvkey': str})

size_raw = panel['size']
age_raw  = panel['age']

# ============================================================
# Original SA (fixed Hadlock-Pierce caps) — recomputed here for
# an apples-to-apples diagnostic comparison against the panel's
# stored 'sa_index' / 'fin_constrained'.
# ============================================================
ORIG_SIZE_CAP = np.log(4500)
ORIG_AGE_CAP  = 37

size_sa_orig = size_raw.clip(upper=ORIG_SIZE_CAP)
age_sa_orig  = age_raw.clip(upper=ORIG_AGE_CAP)
sa_orig = (-0.737 * size_sa_orig
           + 0.043 * size_sa_orig**2
           - 0.040 * age_sa_orig)

orig_thr = sa_orig.quantile(2 / 3)
constrained_orig = (sa_orig >= orig_thr).astype(int)

# ============================================================
# Recalibrated SA — caps at the 95th percentile WITHIN sample
# ============================================================
size_cap_recal = size_raw.quantile(0.95)
age_cap_recal  = age_raw.quantile(0.95)

size_sa_recal = size_raw.clip(upper=size_cap_recal)
age_sa_recal  = age_raw.clip(upper=age_cap_recal)
sa_recal = (-0.737 * size_sa_recal
            + 0.043 * size_sa_recal**2
            - 0.040 * age_sa_recal)

recal_thr = sa_recal.quantile(2 / 3)
constrained_recal = (sa_recal >= recal_thr).astype(int)

panel['sa_index_recalibrated']    = sa_recal
panel['constrained_recalibrated'] = constrained_recal

# ============================================================
# Diagnostics
# ============================================================
n_distinct_orig  = sa_orig.nunique()
n_distinct_recal = sa_recal.nunique()

pct_both_caps_orig = (
    (size_raw >= ORIG_SIZE_CAP) & (age_raw >= ORIG_AGE_CAP)
).mean() * 100
pct_both_caps_recal = (
    (size_raw >= size_cap_recal) & (age_raw >= age_cap_recal)
).mean() * 100

corr_dummies = panel['fin_constrained'].corr(panel['constrained_recalibrated'])
corr_orig_recomputed = constrained_orig.corr(panel['fin_constrained'])

lines = []
lines.append("SA INDEX RECALIBRATION DIAGNOSTICS")
lines.append("=" * 70)
lines.append("")
lines.append(f"Sample: {len(panel)} firm-years, {panel['gvkey'].nunique()} firms")
lines.append("")
lines.append("Caps used:")
lines.append(f"  Original (Hadlock-Pierce):  size cap = log(4500) = {ORIG_SIZE_CAP:.4f}, "
              f"age cap = {ORIG_AGE_CAP}")
lines.append(f"  Recalibrated (95th pctile): size cap = {size_cap_recal:.4f}, "
              f"age cap = {age_cap_recal:.4f}")
lines.append("")
lines.append("Distinct SA values:")
lines.append(f"  Original SA index:      {n_distinct_orig}")
lines.append(f"  Recalibrated SA index:  {n_distinct_recal}")
lines.append("")
lines.append("% of firm-years hitting BOTH caps simultaneously:")
lines.append(f"  Original approach:      {pct_both_caps_orig:.1f}%")
lines.append(f"  Recalibrated approach:  {pct_both_caps_recal:.1f}%")
lines.append("")
lines.append("Sanity check — original sa_index recomputed here vs. panel's stored sa_index "
             "(should be ~identical):")
lines.append(f"  corr(sa_orig recomputed, panel['sa_index']) = "
              f"{sa_orig.corr(panel['sa_index']):.4f}")
lines.append(f"  corr(fin_constrained recomputed, panel['fin_constrained']) = "
              f"{corr_orig_recomputed:.4f}")
lines.append("")
lines.append("Correlation between original and recalibrated constrained dummies:")
lines.append(f"  corr(fin_constrained, constrained_recalibrated) = {corr_dummies:.4f}")
lines.append("")
lines.append("Constrained-dummy cross-tab (original x recalibrated):")
crosstab = pd.crosstab(panel['fin_constrained'], panel['constrained_recalibrated'],
                       rownames=['fin_constrained (orig)'],
                       colnames=['constrained_recalibrated'])
lines.append(crosstab.to_string())
lines.append("")

diagnostics_text = "\n".join(lines)
print(diagnostics_text)

with open('output/sa_recalibration_diagnostics.txt', 'w') as f:
    f.write(diagnostics_text)
print("\nSaved: output/sa_recalibration_diagnostics.txt")

# ============================================================
# Save the recalibrated columns (merge key: gvkey, fyear) for
# use by the H2 robustness regression script.
# ============================================================
out_cols = ['gvkey', 'fyear', 'sa_index_recalibrated', 'constrained_recalibrated']
panel[out_cols].to_csv('data/processed_extended/sa_recalibrated.csv', index=False)
print("Saved: data/processed_extended/sa_recalibrated.csv")
