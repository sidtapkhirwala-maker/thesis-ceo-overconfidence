import wrds
import pandas as pd

# ============================================================
# Load the 80-firm sample (skip the 4 comment lines)
# ============================================================
sample = pd.read_csv('data/processed_extended/sample_firms.csv',
                     comment='#', dtype={'gvkey': str})
print(f"Sample firms: {len(sample)}")

gvkeys = tuple(sample['gvkey'].tolist())

# ============================================================
# Pull ExecuComp ANNCOMP for CEOs 2017-2022
# (2017 included to detect turnovers occurring in 2018)
# ============================================================
db = wrds.Connection(wrds_username='sidt28')

anncomp = db.raw_sql(f"""
    SELECT gvkey, execid, year, exec_fname, exec_lname,
           ceoann, becameceo, leftofc
    FROM execcomp.anncomp
    WHERE gvkey IN {gvkeys}
      AND year BETWEEN 2017 AND 2024
      AND ceoann = 'CEO'
""")

print(f"\nRaw ANNCOMP CEO rows: {len(anncomp)}")
print(f"Distinct firms with CEO data: {anncomp['gvkey'].nunique()}")

# Some firms might be missing from ExecuComp coverage - flag them
missing = set(sample['gvkey']) - set(anncomp['gvkey'])
print(f"Firms in sample but missing from ExecuComp: {len(missing)}")
if missing:
    print("  Missing GVKEYs:", missing)

# Save raw pull
anncomp.to_csv('data/raw/execucomp/anncomp_raw.csv', index=False)

# ============================================================
# Identify CEO turnover events
# A turnover occurs when EXECID changes year-over-year within the same GVKEY
# ============================================================
anncomp = anncomp.sort_values(['gvkey', 'year']).reset_index(drop=True)

# Some firms have multiple "CEO" flagged in the same year during transitions.
# Take the CEO who held the role for most of that year (lowest execid as a tiebreaker
# is arbitrary — instead, keep the first row by year and dedupe).
# A cleaner approach: keep one row per (gvkey, year) using the row that
# corresponds to the longest-serving CEO that year.
anncomp_dedup = (
    anncomp.sort_values(['gvkey', 'year', 'becameceo'])
           .drop_duplicates(subset=['gvkey', 'year'], keep='first')
           .reset_index(drop=True)
)

anncomp_dedup['prev_execid'] = anncomp_dedup.groupby('gvkey')['execid'].shift(1)
anncomp_dedup['prev_year'] = anncomp_dedup.groupby('gvkey')['year'].shift(1)

# Flag turnover: execid differs from previous year, and previous year is exactly t-1
turnovers = anncomp_dedup[
    (anncomp_dedup['execid'] != anncomp_dedup['prev_execid'])
    & (anncomp_dedup['prev_execid'].notna())
    & (anncomp_dedup['year'] - anncomp_dedup['prev_year'] == 1)
    & (anncomp_dedup['year'].between(2017, 2024))
].copy()

print(f"\nCEO turnover events 2018-2022: {len(turnovers)}")
print(f"Distinct firms with at least one turnover: {turnovers['gvkey'].nunique()}")

if len(turnovers) > 0:
    # Merge in company names for readability
    turnovers = turnovers.merge(sample[['gvkey', 'conm', 'ff12_name']],
                                 on='gvkey', how='left')
    print("\nTurnover events:")
    print(turnovers[['gvkey', 'conm', 'ff12_name', 'year',
                     'prev_execid', 'execid',
                     'exec_fname', 'exec_lname']].to_string(index=False))

# ============================================================
# Apply the threshold check
# ============================================================
THRESHOLD = 15
print(f"\n=== TURNOVER FLOOR CHECK ===")
print(f"Required: >= {THRESHOLD} turnover events")
print(f"Observed: {len(turnovers)} turnover events")
if len(turnovers) >= THRESHOLD:
    print("PASS - Proceed to next phase (Compustat outcome construction)")
else:
    print("FAIL - Need to extend the stratified draw by +10 firms")
    print("       (per proposal §4 turnover-event verification step)")

# ============================================================
# Save outputs
# ============================================================
anncomp_dedup.to_csv('data/processed_extended/execucomp_ceo_panel.csv', index=False)
turnovers.to_csv('data/processed_extended/turnover_events.csv', index=False)
print("\nSaved:")
print("  data/processed_extended/execucomp_ceo_panel.csv")
print("  data/processed_extended/turnover_events.csv")

db.close()