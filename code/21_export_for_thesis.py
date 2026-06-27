import pandas as pd
import numpy as np
import os

os.makedirs('output/for_thesis', exist_ok=True)

# ============================================================
# 1) Sample composition
# ============================================================
sample = pd.read_csv('data/processed/sample_firms.csv', comment='#',
                     dtype={'gvkey': str})
print("=" * 70)
print("SAMPLE COMPOSITION")
print("=" * 70)
print(f"Total firms: {len(sample)}")
print("\nBy FF12 industry:")
print(sample['ff12_name'].value_counts().to_string())
sample[['gvkey', 'conm', 'ff12_name', 'sic']].to_csv(
    'output/for_thesis/sample_firms_list.csv', index=False)

# ============================================================
# 2) Regression panel descriptive stats
# ============================================================
panel = pd.read_csv('data/processed/regression_panel.csv',
                    dtype={'gvkey': str})
print("\n" + "=" * 70)
print("REGRESSION PANEL")
print("=" * 70)
print(f"Total firm-year rows: {len(panel)}")
print(f"Distinct firms: {panel['gvkey'].nunique()}")
print(f"H1 sample (book_leverage_t1 non-missing): {panel['book_leverage_t1'].notna().sum()}")
print(f"H2 sample (debt_share_t1 non-missing, gross issuances): {panel['debt_share_t1'].notna().sum()}")
if 'debt_share_strict_t1' in panel.columns:
    print(f"H2 strict-net sample: {panel['debt_share_strict_t1'].notna().sum()}")

cols = ['overconfidence', 'overconfidence_pca', 'overconfidence_2comp',
        'z_net_tone', 'z_strong_modality', 'z_inverse_hedging',
        'book_leverage_t1', 'debt_share_t1',
        'size', 'profitability', 'mtb', 'tangibility', 'cash', 'age',
        'fin_constrained', 'sa_index']
cols = [c for c in cols if c in panel.columns]
desc = panel[cols].describe(percentiles=[0.25, 0.5, 0.75]).round(4)
print("\nDescriptive stats:")
print(desc.to_string())
desc.to_csv('output/for_thesis/descriptive_stats.csv')

# ============================================================
# 3) Component correlation matrix
# ============================================================
comp_cols = ['z_net_tone', 'z_strong_modality', 'z_inverse_hedging']
corr = panel[comp_cols].corr().round(3)
print("\n" + "=" * 70)
print("LM COMPONENT CORRELATION MATRIX")
print("=" * 70)
print(corr.to_string())
corr.to_csv('output/for_thesis/component_correlations.csv')

# ============================================================
# 4) Overconfidence by industry
# ============================================================
ov_by_ind = panel.merge(sample[['gvkey', 'ff12_name']], on='gvkey', how='left')
ind_stats = ov_by_ind.groupby('ff12_name')['overconfidence'].agg(
    ['count', 'mean', 'std', 'min', 'max']).round(3)
print("\n" + "=" * 70)
print("OVERCONFIDENCE INDEX BY INDUSTRY")
print("=" * 70)
print(ind_stats.to_string())
ind_stats.to_csv('output/for_thesis/overconfidence_by_industry.csv')

# ============================================================
# 5) Year coverage
# ============================================================
year_cov = panel.groupby('fyear').size()
print("\n" + "=" * 70)
print("YEAR COVERAGE")
print("=" * 70)
print(year_cov.to_string())

# ============================================================
# 6) Turnover events summary
# ============================================================
turns = pd.read_csv('data/processed/turnover_events.csv', dtype={'gvkey': str})
print("\n" + "=" * 70)
print("CEO TURNOVER EVENTS")
print("=" * 70)
print(f"Total events 2018-2022: {len(turns)}")
print(f"Distinct firms with turnover: {turns['gvkey'].nunique()}")
print(f"\nBy year:")
print(turns['year'].value_counts().sort_index().to_string())

# ============================================================
# 7) Corpus / transcript stats
# ============================================================
manifest = pd.read_csv('data/processed/ceo_speech_manifest.csv',
                       dtype={'gvkey': str})
print("\n" + "=" * 70)
print("CEO SPEECH CORPUS")
print("=" * 70)
print(f"Total extractions: {len(manifest)}")
print(f"Mean CEO words per call: {manifest['total_words'].mean():.0f}")
print(f"Median CEO words per call: {manifest['total_words'].median():.0f}")
print(f"Min / max words: {manifest['total_words'].min()} / {manifest['total_words'].max()}")

# ============================================================
# 8) Random seed
# ============================================================
with open('random_seed.txt') as f:
    print("\n" + "=" * 70)
    print("RANDOM SEED")
    print("=" * 70)
    print(f.read())

print("\n" + "=" * 70)
print("All summary files saved to output/for_thesis/")
print("=" * 70)