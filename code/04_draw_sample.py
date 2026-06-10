import pandas as pd
import numpy as np

# ============================================================
# Fix the random seed and record it
# ============================================================
RANDOM_SEED = 20260608  # today's date as a memorable seed - DO NOT CHANGE
rng = np.random.default_rng(seed=RANDOM_SEED)

with open('random_seed.txt', 'w') as f:
    f.write(f"Stratified sample random seed: {RANDOM_SEED}\n")
    f.write("Used in code/04_draw_sample.py to draw 80-firm sample.\n")

# ============================================================
# Load eligible universe + Compustat panel
# ============================================================
eligible = pd.read_csv('data/processed/universe_eligible.csv', dtype={'gvkey': str})
funda = pd.read_csv('data/processed/compustat_panel.csv', dtype={'gvkey': str})

print(f"Eligible firms: {len(eligible)}")

# ============================================================
# Compute 2017 fiscal-year-end market cap per firm
# Market cap = prcc_f * csho (millions, in Compustat units)
# ============================================================
mcap_2017 = funda[funda['fyear'] == 2017].copy()
mcap_2017['mcap'] = mcap_2017['prcc_f'] * mcap_2017['csho']

# Some eligible firms may not have a 2017 row (filter only required 2018-2022).
# For those, fall back to 2018 fiscal-year-end market cap.
mcap_2018 = funda[funda['fyear'] == 2018].copy()
mcap_2018['mcap'] = mcap_2018['prcc_f'] * mcap_2018['csho']

mcap = (
    pd.concat([mcap_2017[['gvkey', 'mcap']],
               mcap_2018[['gvkey', 'mcap']]])
      .dropna()
      .drop_duplicates(subset='gvkey', keep='first')
)

eligible = eligible.merge(mcap, on='gvkey', how='left')
print(f"Firms with market cap: {eligible['mcap'].notna().sum()} of {len(eligible)}")

# Drop any firm without a market cap (shouldn't happen, but just in case)
eligible = eligible.dropna(subset=['mcap']).copy()

# ============================================================
# Compute industry weights based on aggregate market cap
# ============================================================
ind_totals = eligible.groupby('ff12_name')['mcap'].sum()
total_mcap = ind_totals.sum()
ind_weights = ind_totals / total_mcap

# Compute target firm count per industry, proportional to mcap weight
TARGET_SAMPLE_SIZE = 80
ind_targets_raw = ind_weights * TARGET_SAMPLE_SIZE
ind_targets = ind_targets_raw.round().astype(int)

# Ensure every industry gets at least 1 firm
ind_targets = ind_targets.clip(lower=1)

# Adjust to sum exactly to TARGET_SAMPLE_SIZE
diff = TARGET_SAMPLE_SIZE - ind_targets.sum()
if diff != 0:
    # Add/subtract from the industries with largest fractional remainder
    fractional = (ind_targets_raw - ind_targets_raw.round()).abs()
    adj_order = fractional.sort_values(ascending=False).index.tolist()
    i = 0
    while ind_targets.sum() != TARGET_SAMPLE_SIZE:
        ind = adj_order[i % len(adj_order)]
        if ind_targets.sum() < TARGET_SAMPLE_SIZE:
            ind_targets[ind] += 1
        elif ind_targets[ind] > 1:
            ind_targets[ind] -= 1
        i += 1

print("\nIndustry weights and per-industry sample targets:")
summary = pd.DataFrame({
    'mcap_weight_pct': (ind_weights * 100).round(2),
    'firms_in_pool': eligible.groupby('ff12_name').size(),
    'sample_target': ind_targets
}).sort_values('mcap_weight_pct', ascending=False)
print(summary)
print(f"\nTotal target firms: {ind_targets.sum()}")

# ============================================================
# Draw the sample
# ============================================================
sampled_dfs = []
for industry, n_target in ind_targets.items():
    pool = eligible[eligible['ff12_name'] == industry]
    if n_target > len(pool):
        n_target = len(pool)  # safety: cap at pool size
    drawn = pool.sample(n=n_target, random_state=rng.integers(0, 1_000_000))
    sampled_dfs.append(drawn)

sample = pd.concat(sampled_dfs).sort_values(['ff12_name', 'conm'])
print(f"\nFinal sample size: {len(sample)} firms")

print("\nFinal sample by industry:")
print(sample['ff12_name'].value_counts())

# ============================================================
# Save sample with seed recorded in the header
# ============================================================
with open('data/processed/sample_firms.csv', 'w') as f:
    f.write(f"# Stratified random sample of S&P 500 firms\n")
    f.write(f"# Random seed: {RANDOM_SEED}\n")
    f.write(f"# Sample size: {len(sample)}\n")
    f.write(f"# Stratification: FF12 industry, proportional to 2017 market cap\n")
sample.to_csv('data/processed/sample_firms.csv', mode='a', index=False)

print("\nSaved sample to data/processed/sample_firms.csv")
print(f"Seed recorded in random_seed.txt: {RANDOM_SEED}")