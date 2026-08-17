import wrds
import pandas as pd
from sqlalchemy import text
from rapidfuzz import fuzz
import os
import time

db = wrds.Connection(wrds_username='sidt28')

def safe_sql(query):
    with db.engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
        cols = list(result.keys())
    return pd.DataFrame(rows, columns=cols)

# ============================================================
# Load inputs
# ============================================================
inventory = pd.read_csv('data/processed_extended/transcript_inventory.csv',
                        dtype={'gvkey': str})
ceo_panel = pd.read_csv('data/processed_extended/execucomp_ceo_panel.csv',
                        dtype={'gvkey': str})

print(f"Calls to process: {len(inventory)}")
print(f"CEO panel rows: {len(ceo_panel)}")

# Map: (gvkey, calendar_year) -> list of CEOs that year
# We use call_year as the year to match a CEO, which is the calendar year of the call.
# This handles fiscal-year mismatches by using the simple rule "who was CEO when the call happened?"
ceo_panel['ceo_full_name'] = (
    ceo_panel['exec_fname'].astype(str).str.strip() + ' '
    + ceo_panel['exec_lname'].astype(str).str.strip()
)

# Build a lookup: (gvkey, year) -> list of candidate CEO names
ceo_lookup = (
    ceo_panel.groupby(['gvkey', 'year'])['ceo_full_name']
             .apply(list)
             .to_dict()
)

# Make sure output directory exists
os.makedirs('data/processed_extended/ceo_speech', exist_ok=True)

# ============================================================
# Loop over transcripts: identify CEO, extract speech
# ============================================================
manifest = []
unmatched = []

start = time.time()
for i, row in inventory.iterrows():
    tid = int(row['transcriptid'])
    gvkey = row['gvkey']
    call_year = int(row['call_year'])
    call_q = int(row['call_quarter'])
    call_date = row['mostimportantdateutc']
    conm = row.get('companyname', '')

    # ----- 1. Who is the CEO during this call? -----
    # Try call_year first; if not in ExecuComp for that year, try year-1 or year+1.
    ceo_names = (
        ceo_lookup.get((gvkey, call_year))
        or ceo_lookup.get((gvkey, call_year - 1))
        or ceo_lookup.get((gvkey, call_year + 1))
        or []
    )
    if not ceo_names:
        unmatched.append({**row.to_dict(), 'reason': 'no_ceo_in_execucomp_window'})
        continue

    # ----- 2. Get all Executive-type speakers on this transcript -----
    speakers = safe_sql(f"""
        SELECT DISTINCT p.transcriptpersonid, p.transcriptpersonname
        FROM ciq_transcripts.ciqtranscriptcomponent c
        JOIN ciq_transcripts.ciqtranscriptperson p
          ON c.transcriptpersonid = p.transcriptpersonid
        WHERE c.transcriptid = {tid}
          AND p.speakertypeid = 2
    """)
    if len(speakers) == 0:
        unmatched.append({**row.to_dict(), 'reason': 'no_executives_on_transcript'})
        continue

    # ----- 3. Fuzzy match CEO name(s) against executive speakers -----
    best_match = None
    best_score = 0
    best_ceo = None
    for cname in ceo_names:
        for _, spk in speakers.iterrows():
            sname = str(spk['transcriptpersonname'])
            score = fuzz.token_set_ratio(cname.lower(), sname.lower())
            if score > best_score:
                best_score = score
                best_match = spk
                best_ceo = cname

    if best_score < 85:
        unmatched.append({**row.to_dict(),
                          'reason': f'low_match_score ({best_score})',
                          'best_speaker': best_match['transcriptpersonname'] if best_match is not None else None,
                          'ceo_candidates': '|'.join(ceo_names)})
        continue

    # ----- 4. Pull CEO's Presenter Speech + Answer segments -----
    ceo_pid = int(best_match['transcriptpersonid'])
    segs = safe_sql(f"""
        SELECT componentorder, transcriptcomponenttypeid, componenttext
        FROM ciq_transcripts.ciqtranscriptcomponent
        WHERE transcriptid = {tid}
          AND transcriptpersonid = {ceo_pid}
          AND transcriptcomponenttypeid IN (2, 4)
        ORDER BY componentorder
    """)

    full = "\n\n".join(segs['componenttext'].astype(str).tolist())
    word_count = len(full.split())

    # ----- 5. Save speech file + manifest entry -----
    fname = f"{gvkey}_{call_year}_Q{call_q}_{tid}.txt"
    fpath = f"data/processed_extended/ceo_speech/{fname}"
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(full)

    manifest.append({
        'gvkey': gvkey,
        'conm': conm,
        'call_year': call_year,
        'call_quarter': call_q,
        'call_date': call_date,
        'transcriptid': tid,
        'ceo_matched': best_ceo,
        'speaker_matched': best_match['transcriptpersonname'],
        'match_score': best_score,
        'n_presenter_segments': int((segs['transcriptcomponenttypeid']==2).sum()),
        'n_answer_segments': int((segs['transcriptcomponenttypeid']==4).sum()),
        'total_words': word_count,
        'filename': fname,
    })

    # Progress every 50 calls
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        rate = (i + 1) / elapsed
        eta = (len(inventory) - i - 1) / rate
        print(f"  {i+1}/{len(inventory)} processed | "
              f"{len(manifest)} extracted, {len(unmatched)} unmatched | "
              f"ETA {eta/60:.1f} min")

print(f"\nDone: {len(manifest)} extractions, {len(unmatched)} unmatched")
print(f"Elapsed: {(time.time()-start)/60:.1f} min")

# ============================================================
# Save manifest and unmatched log
# ============================================================
pd.DataFrame(manifest).to_csv('data/processed_extended/ceo_speech_manifest.csv', index=False)
pd.DataFrame(unmatched).to_csv('data/processed_extended/ceo_speech_unmatched.csv', index=False)

print("\nSaved:")
print("  data/processed_extended/ceo_speech_manifest.csv")
print("  data/processed_extended/ceo_speech_unmatched.csv")
print(f"  data/processed_extended/ceo_speech/  ({len(manifest)} text files)")

db.close()