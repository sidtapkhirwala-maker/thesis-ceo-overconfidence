import wrds
import pandas as pd
from sqlalchemy import text
from rapidfuzz import fuzz
import re
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
# Name normalisation: strip credentials, suffixes, initials
# ============================================================
# Words to strip from names (case-insensitive). Simpler and safer than regex.
CREDENTIAL_TOKENS = {
    'jr', 'sr', 'ii', 'iii', 'iv', 'v',
    'phd', 'jd', 'md', 'mba', 'cpa', 'bsc', 'ba', 'cfa', 'esq',
    'hons', 'hon',
}

def normalise_name(name):
    if not isinstance(name, str):
        return ''
    n = name.lower()
    # Strip punctuation and parentheses
    n = re.sub(r'[.,;()\$$\$$]', ' ', n)
    # Tokenise, drop credential tokens, collapse spaces
    tokens = [t for t in n.split() if t not in CREDENTIAL_TOKENS]
    return ' '.join(tokens)

def last_name(name):
    tokens = [t for t in normalise_name(name).split() if len(t) > 1]
    return tokens[-1] if tokens else ''

def first_name(name):
    tokens = [t for t in normalise_name(name).split() if len(t) > 1]
    return tokens[0] if tokens else ''

# ============================================================
# Load unmatched calls and ExecuComp CEO panel
# ============================================================
unm = pd.read_csv('data/processed/ceo_speech_unmatched.csv',
                  dtype={'gvkey': str})
ceo_panel = pd.read_csv('data/processed/execucomp_ceo_panel.csv',
                        dtype={'gvkey': str})
ceo_panel['ceo_full_name'] = (
    ceo_panel['exec_fname'].astype(str).str.strip() + ' '
    + ceo_panel['exec_lname'].astype(str).str.strip()
)
ceo_lookup = (
    ceo_panel.groupby(['gvkey', 'year'])['ceo_full_name']
             .apply(list)
             .to_dict()
)

print(f"Unmatched calls to retry: {len(unm)}")

os.makedirs('data/processed/ceo_speech', exist_ok=True)

# ============================================================
# Retry loop with improved matching
# ============================================================
recovered = []
still_unmatched = []

start = time.time()
for i, row in unm.iterrows():
    tid = int(row['transcriptid'])
    gvkey = row['gvkey']
    call_year = int(row['call_year'])
    call_q = int(row['call_quarter'])
    call_date = row['mostimportantdateutc']
    conm = row.get('companyname', '')

    # Get CEO candidates for this firm-year
    ceo_names = (
        ceo_lookup.get((gvkey, call_year))
        or ceo_lookup.get((gvkey, call_year - 1))
        or ceo_lookup.get((gvkey, call_year + 1))
        or []
    )
    if not ceo_names:
        still_unmatched.append({**row.to_dict(), 'rescue_reason': 'no_ceo'})
        continue

    # Pull executive speakers on this transcript
    speakers = safe_sql(f"""
        SELECT DISTINCT p.transcriptpersonid, p.transcriptpersonname
        FROM ciq_transcripts.ciqtranscriptcomponent c
        JOIN ciq_transcripts.ciqtranscriptperson p
          ON c.transcriptpersonid = p.transcriptpersonid
        WHERE c.transcriptid = {tid}
          AND p.speakertypeid = 2
    """)
    if len(speakers) == 0:
        still_unmatched.append({**row.to_dict(), 'rescue_reason': 'no_executives'})
        continue

    # Normalise everything
    speakers['norm_name'] = speakers['transcriptpersonname'].apply(normalise_name)
    speakers['last'] = speakers['transcriptpersonname'].apply(last_name)
    speakers['first'] = speakers['transcriptpersonname'].apply(first_name)

    # Score each (CEO candidate, speaker) pair using a composite rule
    best = None
    best_score = 0
    best_ceo = None
    best_method = None
    for cname in ceo_names:
        c_norm = normalise_name(cname)
        c_last = last_name(cname)
        c_first = first_name(cname)
        for _, spk in speakers.iterrows():
            # 1. Normalised full-name fuzzy match
            s_full = fuzz.token_set_ratio(c_norm, spk['norm_name'])
            # 2. Last-name exact match (very strong signal if it agrees)
            s_last = 100 if c_last and c_last == spk['last'] else 0
            # 3. First letter of first name + last name (handles "R. Norwitt" vs "Richard Norwitt")
            s_initial = 0
            if c_last == spk['last'] and c_first and spk['first']:
                if c_first[0] == spk['first'][0]:
                    s_initial = 95
            score = max(s_full, s_last, s_initial)
            if score > best_score:
                best_score = score
                best = spk
                best_ceo = cname
                best_method = (
                    'last_exact' if score == s_last and s_last == 100 else
                    'first_initial+last' if score == s_initial else
                    'fuzzy'
                )

    if best_score < 90:
        still_unmatched.append({**row.to_dict(),
                                'rescue_reason': f'still_low ({best_score:.1f})',
                                'best_speaker': best['transcriptpersonname'] if best is not None else None})
        continue

    # Pull the CEO's speech
    ceo_pid = int(best['transcriptpersonid'])
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

    fname = f"{gvkey}_{call_year}_Q{call_q}_{tid}.txt"
    fpath = f"data/processed/ceo_speech/{fname}"
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(full)

    recovered.append({
        'gvkey': gvkey,
        'conm': conm,
        'call_year': call_year,
        'call_quarter': call_q,
        'call_date': call_date,
        'transcriptid': tid,
        'ceo_matched': best_ceo,
        'speaker_matched': best['transcriptpersonname'],
        'match_score': best_score,
        'match_method': best_method,
        'n_presenter_segments': int((segs['transcriptcomponenttypeid']==2).sum()),
        'n_answer_segments': int((segs['transcriptcomponenttypeid']==4).sum()),
        'total_words': word_count,
        'filename': fname,
    })

    if (i + 1) % 25 == 0:
        elapsed = time.time() - start
        print(f"  {i+1}/{len(unm)} retried | recovered {len(recovered)} | still unmatched {len(still_unmatched)}")

print(f"\nRescue complete: recovered {len(recovered)} / {len(unm)} calls")
print(f"Still unmatched: {len(still_unmatched)}")
print(f"Elapsed: {(time.time()-start)/60:.1f} min")

# Append the recovered entries to the existing manifest
existing = pd.read_csv('data/processed/ceo_speech_manifest.csv',
                       dtype={'gvkey': str})
combined = pd.concat([existing, pd.DataFrame(recovered)], ignore_index=True)
combined.to_csv('data/processed/ceo_speech_manifest.csv', index=False)
print(f"\nManifest now contains {len(combined)} total extractions")

pd.DataFrame(still_unmatched).to_csv(
    'data/processed/ceo_speech_still_unmatched.csv', index=False
)
print(f"Still-unmatched log: data/processed/ceo_speech_still_unmatched.csv")

# Quick look at remaining failures
if still_unmatched:
    sdf = pd.DataFrame(still_unmatched)
    print("\nRemaining unmatched by firm (top 10):")
    print(sdf.groupby('gvkey').size().sort_values(ascending=False).head(10).to_string())

db.close()