import pandas as pd

# ============================================================
# Load the manifest
# ============================================================
manifest = pd.read_csv('data/processed/ceo_speech_manifest.csv',
                      dtype={'gvkey': str})
print(f"Total extracted calls: {len(manifest)}")
print(f"Distinct firms: {manifest['gvkey'].nunique()}")

# ============================================================
# Distribution before any filtering
# ============================================================
print(f"\nWord count distribution (all extracted calls):")
print(manifest['total_words'].describe().to_string())

# ============================================================
# Stage 1: drop calls below 1,000-word threshold
# ============================================================
THRESHOLD = 1000
before = len(manifest)
manifest['passes_word_filter'] = manifest['total_words'] >= THRESHOLD
surviving_calls = manifest[manifest['passes_word_filter']].copy()
print(f"\n=== Stage 1: 1,000-word filter ===")
print(f"Passed: {len(surviving_calls)} / {before} ({len(surviving_calls)/before*100:.1f}%)")
print(f"Dropped: {before - len(surviving_calls)} calls below threshold")

# Where do the failures concentrate?
failed = manifest[~manifest['passes_word_filter']]
if len(failed) > 0:
    print(f"\nWord-count distribution of failed calls:")
    print(failed['total_words'].describe().to_string())

# ============================================================
# Stage 2: drop firm-years with fewer than 2 surviving calls
# ============================================================
fy_counts = (
    surviving_calls.groupby(['gvkey', 'call_year'])
                   .size()
                   .reset_index(name='n_surviving_calls')
)
fy_keep = fy_counts[fy_counts['n_surviving_calls'] >= 2]
fy_drop = fy_counts[fy_counts['n_surviving_calls'] < 2]

print(f"\n=== Stage 2: firm-year >= 2 calls filter ===")
print(f"Firm-years that survive: {len(fy_keep)}")
print(f"Firm-years dropped (<2 calls): {len(fy_drop)}")

# Restrict surviving calls to surviving firm-years
final_calls = surviving_calls.merge(
    fy_keep[['gvkey', 'call_year']], on=['gvkey', 'call_year'], how='inner'
)

# ============================================================
# Final panel statistics
# ============================================================
print(f"\n=== FINAL CORPUS ===")
print(f"Total quarterly transcripts: {len(final_calls)}")
print(f"Distinct firms: {final_calls['gvkey'].nunique()}")
print(f"Distinct firm-years: {len(fy_keep)}")
print(f"Of theoretical max (80 firms x 5 years = 400 firm-years): "
      f"{len(fy_keep)/400*100:.1f}%")

print(f"\nFirm-year coverage by year:")
print(fy_keep.groupby('call_year').size().to_string())

print(f"\nQuarterly calls per firm-year distribution:")
print(fy_keep['n_surviving_calls'].value_counts().sort_index().to_string())

# ============================================================
# Save outputs
# ============================================================
final_calls.to_csv('data/processed/ceo_speech_final.csv', index=False)
fy_keep.to_csv('data/processed/final_firm_years.csv', index=False)
fy_drop.to_csv('data/processed/dropped_firm_years.csv', index=False)

print("\nSaved:")
print("  data/processed/ceo_speech_final.csv     <- use this for index construction")
print("  data/processed/final_firm_years.csv     <- use this to filter Compustat panel")
print("  data/processed/dropped_firm_years.csv   <- for the limitations section")