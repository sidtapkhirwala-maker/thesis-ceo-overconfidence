import wrds
import pandas as pd

# ============================================================
# Load the post-exclusion universe
# ============================================================
universe = pd.read_csv('data/processed/universe_after_exclusions.csv', dtype={'gvkey': str})
print(f"Post-exclusion universe: {len(universe)} firms")

gvkeys = tuple(universe['gvkey'].tolist())

# ============================================================
# Pull Compustat annual fundamentals 2017-2023
# (2017 for lagged controls, 2023 for t+1 outcome of 2022 language)
# ============================================================
db = wrds.Connection(wrds_username='sidt28')

funda = db.raw_sql(f"""
    SELECT gvkey, datadate, fyear, at, dltt, dlc, oibdp, ppent, che,
           prcc_f, csho, dltis, dltr, sstk, prstkc, sich
    FROM comp.funda
    WHERE gvkey IN {gvkeys}
      AND fyear BETWEEN 2017 AND 2023
      AND indfmt='INDL' AND datafmt='STD'
      AND popsrc='D' AND consol='C'
""")

print(f"\nRaw Compustat rows pulled: {len(funda)}")
print(f"Distinct firms in Compustat: {funda['gvkey'].nunique()}")

# Save raw pull
funda.to_csv('data/raw/compustat/funda_raw.csv', index=False)

# ============================================================
# Apply the continuous-listing filter
# Keep only firms with non-null key variables in EVERY year 2018-2022
# ============================================================
key_vars = ['at', 'dltt', 'dlc', 'oibdp', 'ppent', 'che', 'prcc_f', 'csho']
sample_years = [2018, 2019, 2020, 2021, 2022]

# Restrict to the 5 core years for the filter check
core = funda[funda['fyear'].isin(sample_years)].copy()

# For each firm: must have all 5 years AND no missing key vars in any of them
def is_complete(group):
    if set(group['fyear']) != set(sample_years):
        return False
    return group[key_vars].notna().all().all()

complete_firms = (
    core.groupby('gvkey')
        .filter(is_complete)['gvkey']
        .unique()
)

print(f"\nFirms passing continuous-listing filter: {len(complete_firms)}")

# Filter the universe down
eligible = universe[universe['gvkey'].isin(complete_firms)].copy()
print(f"\nEligible universe (final pool for sampling): {len(eligible)} firms")

print("\nFF12 industry distribution AFTER continuous-listing filter:")
print(eligible['ff12_name'].value_counts())

# Save outputs
eligible.to_csv('data/processed/universe_eligible.csv', index=False)
funda[funda['gvkey'].isin(complete_firms)].to_csv(
    'data/processed/compustat_panel.csv', index=False
)

print("\nSaved to:")
print("  data/processed/universe_eligible.csv")
print("  data/processed/compustat_panel.csv")

db.close()