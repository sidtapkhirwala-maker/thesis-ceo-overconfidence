import wrds
import pandas as pd
from sqlalchemy import text

db = wrds.Connection(wrds_username='sidt28')

def safe_sql(query):
    with db.engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
        cols = list(result.keys())
    return pd.DataFrame(rows, columns=cols)

# ============================================================
# 1) Load the 80-firm sample and bridge to Capital IQ companyid
# ============================================================
sample = pd.read_csv('data/processed/sample_firms.csv',
                     comment='#', dtype={'gvkey': str})
print(f"Sample firms: {len(sample)}")

gvkeys = tuple(sample['gvkey'].tolist())

bridge = safe_sql(f"""
    SELECT gvkey, companyid, companyname, primaryflag, startdate, enddate
    FROM ciq.wrds_gvkey
    WHERE gvkey IN {gvkeys}
""")
print(f"\nBridge rows pulled: {len(bridge)}")

# Some firms may have multiple companyid rows (e.g., post-merger). Keep primaryflag=1.
bridge_primary = bridge[bridge['primaryflag'] == 1].copy()

# Some firms may still lack a primary; fall back to any row for them
firms_with_primary = set(bridge_primary['gvkey'])
firms_missing_primary = set(sample['gvkey']) - firms_with_primary
if firms_missing_primary:
    fallback = bridge[bridge['gvkey'].isin(firms_missing_primary)].drop_duplicates('gvkey')
    bridge_primary = pd.concat([bridge_primary, fallback])

# Final check
firms_with_cid = set(bridge_primary['gvkey'])
missing = set(sample['gvkey']) - firms_with_cid
print(f"Firms successfully bridged: {len(firms_with_cid)} of {len(sample)}")
if missing:
    print(f"  Missing: {missing}")

bridge_primary[['gvkey', 'companyid', 'companyname']].to_csv(
    'data/processed/gvkey_companyid_bridge.csv', index=False
)

# ============================================================
# 2) Pull ALL earnings call transcripts for these companies, 2018-2022
# ============================================================
companyids = tuple(bridge_primary['companyid'].astype(int).tolist())

print(f"\nPulling all earnings call transcripts for {len(companyids)} firms (2018-2022)...")
transcripts = safe_sql(f"""
    SELECT d.companyid,
           d.transcriptid,
           d.mostimportantdateutc,
           d.headline,
           d.transcriptcollectiontypename,
           d.transcriptpresentationtypename,
           d.transcriptcreationdate_utc,
           d.audiolengthsec
    FROM ciq.wrds_transcript_detail d
    WHERE d.companyid IN {companyids}
      AND d.keydeveventtypeid = 48
      AND d.mostimportantdateutc BETWEEN '2018-01-01' AND '2022-12-31'
""")
print(f"Raw transcript-version rows: {len(transcripts)}")

# ============================================================
# 3) Choose ONE canonical transcript per (companyid, call-date)
#    Priority: Final > Preliminary, then most recent creationdate.
# ============================================================
def presentation_priority(p):
    return {'Final': 0, 'Preliminary': 1}.get(p, 2)

transcripts['priority'] = transcripts['transcriptpresentationtypename'].apply(presentation_priority)
transcripts_sorted = transcripts.sort_values(
    ['companyid', 'mostimportantdateutc', 'priority', 'transcriptcreationdate_utc'],
    ascending=[True, True, True, False]
)
canonical = transcripts_sorted.drop_duplicates(
    subset=['companyid', 'mostimportantdateutc'], keep='first'
).copy()

print(f"\nCanonical transcripts (one per call-date): {len(canonical)}")

# ============================================================
# 4) Add fiscal year and quarter labels from the headline
#    (Capital IQ headlines look like "Apple Inc., Q4 2022 Earnings Call, ...")
# ============================================================
import re
def parse_quarter_year(headline):
    if not isinstance(headline, str):
        return (None, None)
    m = re.search(r'Q([1-4])\s+(\d{4})', headline)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (None, None)

canonical[['call_quarter', 'call_year']] = canonical['headline'].apply(
    lambda h: pd.Series(parse_quarter_year(h))
)

# Restrict to calls clearly labelled with Q1-Q4 and year 2018-2022
canonical = canonical[
    canonical['call_year'].between(2018, 2022)
    & canonical['call_quarter'].between(1, 4)
].copy()

# Merge in gvkey for cross-referencing
canonical = canonical.merge(
    bridge_primary[['companyid', 'gvkey', 'companyname']],
    on='companyid', how='left'
)

print(f"\nCanonical earnings calls with Q/Y labels: {len(canonical)}")
print(f"Distinct firms with at least one call: {canonical['gvkey'].nunique()}")

# ============================================================
# 5) Coverage diagnostic: how many calls per (gvkey, year)?
# ============================================================
coverage = (
    canonical.groupby(['gvkey', 'call_year'])
             .size()
             .reset_index(name='n_calls')
)
print("\nCoverage distribution (calls per firm-year):")
print(coverage['n_calls'].value_counts().sort_index())

firmyears_with_4 = (coverage['n_calls'] == 4).sum()
firmyears_with_lt2 = (coverage['n_calls'] < 2).sum()
print(f"\nFirm-years with all 4 quarterly calls: {firmyears_with_4}")
print(f"Firm-years with fewer than 2 calls: {firmyears_with_lt2}")

# ============================================================
# 6) Save outputs
# ============================================================
canonical.to_csv('data/processed/transcript_inventory.csv', index=False)
coverage.to_csv('data/processed/transcript_coverage.csv', index=False)
print("\nSaved:")
print("  data/processed/gvkey_companyid_bridge.csv")
print("  data/processed/transcript_inventory.csv")
print("  data/processed/transcript_coverage.csv")

db.close()