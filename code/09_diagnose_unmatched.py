import pandas as pd

unm = pd.read_csv('data/processed/ceo_speech_unmatched.csv',
                  dtype={'gvkey': str})
print(f"Total unmatched: {len(unm)}")

# Add a cleaner "score" column from the reason string
unm['score'] = unm['reason'].str.extract(r'$$$([\d.]+)$$$').astype(float)

# How many unmatched per firm?
print("\nUnmatched per firm (top 20):")
firm_counts = unm.groupby(['gvkey']).size().sort_values(ascending=False).head(20)
print(firm_counts.to_string())

# Show 30 example unmatched cases so we can eyeball the mismatch patterns
print("\n=== Sample of 30 unmatched calls (CEO candidates vs. best speaker found) ===")
sample_cols = ['gvkey', 'call_year', 'call_quarter', 'score',
               'ceo_candidates', 'best_speaker']
print(unm[sample_cols].head(30).to_string(index=False))

# How many would we recover at score >= 70 vs. >= 60?
print("\n=== Score distribution ===")
for thr in [60, 65, 70, 75, 80, 85]:
    n_above = (unm['score'] >= thr).sum()
    print(f"  Score >= {thr}: {n_above} ({n_above/len(unm)*100:.1f}%)")