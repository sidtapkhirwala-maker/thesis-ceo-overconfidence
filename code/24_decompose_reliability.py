"""
Decompose the within-firm-year reliability of 0.862 into:
  (1) between-firm variance     (absorbed by firm fixed effects)
  (2) within-firm between-year variance  (the identifying variation in FE regressions)
  (3) within-firm-year quarterly noise   (measurement error, reduced by averaging K=3.72)

Reliability of the within-firm year-to-year wiggle = (2) / [(2) + (3)/K]
This is the right reliability for the latent-trait MDE inflation under firm FE.
"""
import pandas as pd
import numpy as np

calls = pd.read_csv('data/processed_extended/call_scores.csv',
                    dtype={'gvkey': str})

# Standardise the three component scores
for col in ['net_tone', 'strong_modality', 'inverse_hedging']:
    calls[f'z_{col}'] = (calls[col] - calls[col].mean()) / calls[col].std()
calls['composite_call'] = calls[['z_net_tone',
                                  'z_strong_modality',
                                  'z_inverse_hedging']].mean(axis=1)

calls['fyear'] = calls['call_year']

# Firm-year means (the level used in non-FE regressions)
fy = calls.groupby(['gvkey', 'fyear'])['composite_call'].mean().reset_index()
fy = fy.rename(columns={'composite_call': 'composite_fy'})

# Firm means (absorbed by firm fixed effects)
firm_means = fy.groupby('gvkey')['composite_fy'].mean().reset_index()
firm_means = firm_means.rename(columns={'composite_fy': 'firm_mean'})

fy = fy.merge(firm_means, on='gvkey', how='left')
fy['within_firm_year'] = fy['composite_fy'] - fy['firm_mean']

# ============================================================
# Variance decomposition
# ============================================================
# (1) Between-firm variance: variance of firm means
between_firm_var = firm_means['firm_mean'].var()
# (2) Within-firm between-year variance: variance of (firm-year mean - firm mean)
within_firm_between_year_var = fy['within_firm_year'].var()
# (3) Within-firm-year quarterly noise (per-call): variance of (per-call composite
#     - firm-year mean), averaged across firm-years
quarterly_var_per_fy = calls.merge(fy[['gvkey', 'fyear', 'composite_fy']],
                                    on=['gvkey', 'fyear'], how='left')
quarterly_var_per_fy['call_dev'] = (quarterly_var_per_fy['composite_call']
                                     - quarterly_var_per_fy['composite_fy'])
within_fy_quarterly_var = (quarterly_var_per_fy.groupby(['gvkey', 'fyear'])['call_dev']
                            .apply(lambda x: x.pow(2).sum() / max(len(x) - 1, 1)).mean())

print("=" * 60)
print("VARIANCE DECOMPOSITION OF FIRM-YEAR COMPOSITE")
print("=" * 60)
print(f"Between-firm variance:                 {between_firm_var:.4f}")
print(f"Within-firm between-year variance:     {within_firm_between_year_var:.4f}")
print(f"Within-firm-year quarterly noise:      {within_fy_quarterly_var:.4f}")
total_var = between_firm_var + within_firm_between_year_var
print(f"\nTotal level variance (1)+(2):           {total_var:.4f}")
print(f"  Between-firm share:                  {between_firm_var/total_var*100:.1f}%")
print(f"  Within-firm between-year share:      {within_firm_between_year_var/total_var*100:.1f}%")

# ============================================================
# Reliability of the level (matches the 0.862 from script 23)
# ============================================================
K = 3.72  # average calls per firm-year
level_signal = total_var
level_noise = within_fy_quarterly_var / K
level_reliability = level_signal / (level_signal + level_noise)
print(f"\nLevel reliability (matches 0.862):     {level_reliability:.4f}")

# ============================================================
# Reliability of the WITHIN-FIRM wiggle (firm FE regression identification)
# ============================================================
# The firm FE absorbs between-firm variance. What's left to identify the regression:
#   signal = within-firm between-year variance
#   noise  = within-firm-year quarterly noise / K
fe_signal = within_firm_between_year_var
fe_noise = within_fy_quarterly_var / K
fe_reliability = fe_signal / (fe_signal + fe_noise)

print("\n" + "=" * 60)
print("RELIABILITY OF THE WITHIN-FIRM YEAR-TO-YEAR WIGGLE")
print("(this is the operative reliability for firm-FE regressions)")
print("=" * 60)
print(f"Signal (within-firm between-year):     {fe_signal:.4f}")
print(f"Noise  (quarterly within-fy / K=3.72): {fe_noise:.4f}")
print(f"FE-identification reliability:         {fe_reliability:.4f}")
print(f"Attenuation factor (1/reliability):    {1/fe_reliability:.4f}")

print("\n" + "=" * 60)
print("LATENT-TRAIT MDE INFLATION UNDER FE-IDENTIFICATION RELIABILITY")
print("=" * 60)
for hyp, measured_mde in [('H1', 0.012), ('H2', 0.059), ('H3', 0.101)]:
    latent_mde = measured_mde / fe_reliability
    print(f"  {hyp}: measured MDE {measured_mde*100:.1f}pp  ->  "
          f"latent-trait MDE {latent_mde*100:.1f}pp")

print("\nFor comparison, level-reliability inflation (what the thesis currently uses):")
for hyp, measured_mde in [('H1', 0.012), ('H2', 0.059), ('H3', 0.101)]:
    latent_mde = measured_mde / level_reliability
    print(f"  {hyp}: measured MDE {measured_mde*100:.1f}pp  ->  "
          f"latent-trait MDE {latent_mde*100:.1f}pp")