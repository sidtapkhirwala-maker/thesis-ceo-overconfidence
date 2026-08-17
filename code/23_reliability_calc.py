"""
Reliability calculation for the linguistic overconfidence index.
Two reliability metrics:
  (1) Within firm-year ICC across the 4 quarterly composite scores
  (2) Cronbach's alpha across the 3 components (net tone, strong modality, inverse hedging)

The latent-trait MDE = measured MDE / reliability.
"""
import pandas as pd
import numpy as np

# Load quarterly call-level scores (one row per call)
calls = pd.read_csv('data/processed_extended/call_scores.csv',
                    dtype={'gvkey': str})
print(f"Quarterly calls in scoring panel: {len(calls)}")
print(f"Columns: {calls.columns.tolist()}")

# Standardise the three component scores across the call-level panel
for col in ['net_tone', 'strong_modality', 'inverse_hedging']:
    calls[f'z_{col}'] = (calls[col] - calls[col].mean()) / calls[col].std()

# Per-call composite score (simple average of three z-scored components)
calls['composite_call'] = calls[['z_net_tone',
                                  'z_strong_modality',
                                  'z_inverse_hedging']].mean(axis=1)

# ============================================================
# RELIABILITY 1: Within firm-year ICC across the 4 quarterly composites
# (signal-to-total variance ratio = between-firm-year variance / total variance)
# ============================================================
# Build (firm-year, call) panel of composite scores
calls['fyear'] = calls['call_year']
fy_groups = calls.groupby(['gvkey', 'fyear'])['composite_call']

# Between-firm-year variance: var of firm-year means
between_var = fy_groups.mean().var()
# Within-firm-year variance: average within-group variance
within_var = fy_groups.var().mean()
# Total variance
total_var = calls['composite_call'].var()

icc = between_var / (between_var + within_var)
print("\n" + "=" * 60)
print("RELIABILITY 1: Within firm-year ICC across quarterly composites")
print("=" * 60)
print(f"Between firm-year variance: {between_var:.4f}")
print(f"Within firm-year variance:  {within_var:.4f}")
print(f"Total variance:             {total_var:.4f}")
print(f"ICC (single-call reliability): {icc:.4f}")
# Spearman-Brown adjusted reliability for averaging K calls
# reliability_K = K * ICC / (1 + (K-1) * ICC)
n_calls_per_fy = fy_groups.size().mean()
print(f"Average calls per firm-year: {n_calls_per_fy:.2f}")
sb_reliability = n_calls_per_fy * icc / (1 + (n_calls_per_fy - 1) * icc)
print(f"Spearman-Brown adjusted reliability (firm-year level): {sb_reliability:.4f}")

# ============================================================
# RELIABILITY 2: Cronbach's alpha across the 3 components
# ============================================================
# Aggregate to firm-year first, then compute alpha across the three components
fy_components = calls.groupby(['gvkey', 'fyear'])[
    ['z_net_tone', 'z_strong_modality', 'z_inverse_hedging']
].mean()
k = 3  # number of components
component_var_sum = fy_components.var().sum()
total_composite_var = fy_components.sum(axis=1).var()
alpha = (k / (k - 1)) * (1 - component_var_sum / total_composite_var)

print("\n" + "=" * 60)
print("RELIABILITY 2: Cronbach's alpha across the 3 components")
print("=" * 60)
print(f"Sum of component variances: {component_var_sum:.4f}")
print(f"Variance of composite sum:  {total_composite_var:.4f}")
print(f"Cronbach's alpha (3 components): {alpha:.4f}")

# ============================================================
# Combined reliability and attenuation factor
# ============================================================
print("\n" + "=" * 60)
print("LATENT-TRAIT MDE INFLATION")
print("=" * 60)
combined_reliability = sb_reliability  # use the firm-year-level reliability
print(f"Firm-year measured-index reliability: {combined_reliability:.4f}")
print(f"Attenuation factor (1/reliability): {1/combined_reliability:.4f}")

print(f"\nMeasured MDE -> Latent-trait MDE inflation:")
for hyp, measured_mde in [('H1', 0.012), ('H2', 0.059), ('H3', 0.101)]:
    latent_mde = measured_mde / combined_reliability
    print(f"  {hyp}: measured MDE {measured_mde*100:.1f}pp  ->  "
          f"latent-trait MDE {latent_mde*100:.1f}pp")

print(f"\nCronbach's alpha-based attenuation factor (cross-component): {1/alpha:.4f}")
print(f"\nLatent-trait MDE under cross-component reliability:")
for hyp, measured_mde in [('H1', 0.012), ('H2', 0.059), ('H3', 0.101)]:
    latent_mde = measured_mde / alpha
    print(f"  {hyp}: measured MDE {measured_mde*100:.1f}pp  ->  "
          f"latent-trait MDE {latent_mde*100:.1f}pp")