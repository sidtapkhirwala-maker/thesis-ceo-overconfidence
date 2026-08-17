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

APPLE_CID = 24937

# ============================================================
# 1) Get all Apple Q4 2022 earnings call transcripts and pick the canonical one
# ============================================================
print("All Apple Q4 2022 earnings call transcripts (with presentation type):")
calls = safe_sql(f"""
    SELECT d.transcriptid,
           d.mostimportantdateutc,
           d.headline,
           d.transcriptcollectiontypename,
           d.transcriptpresentationtypename,
           d.transcriptcreationdate_utc
    FROM ciq.wrds_transcript_detail d
    WHERE d.companyid = {APPLE_CID}
      AND d.keydeveventtypeid = 48
      AND d.mostimportantdateutc = '2022-10-27'
    ORDER BY d.transcriptcreationdate_utc DESC
""")
print(calls.to_string(index=False))

# Pick the most recent "Final" version, otherwise the most recent transcript
if (calls['transcriptpresentationtypename'] == 'Final').any():
    chosen = calls[calls['transcriptpresentationtypename'] == 'Final'].iloc[0]
else:
    chosen = calls.iloc[0]
tid = int(chosen['transcriptid'])
print(f"\nChosen transcriptid: {tid}")

# ============================================================
# 2) Pull all speakers on that call
# ============================================================
print(f"\nAll speakers on transcript {tid}:")
speakers = safe_sql(f"""
    SELECT DISTINCT p.transcriptpersonid,
                    p.transcriptpersonname,
                    p.companyname,
                    p.speakertypeid,
                    s.speakertypename
    FROM ciq_transcripts.ciqtranscriptcomponent c
    JOIN ciq_transcripts.ciqtranscriptperson p
      ON c.transcriptpersonid = p.transcriptpersonid
    LEFT JOIN ciq_transcripts.ciqtranscriptspeakertype s
      ON p.speakertypeid = s.speakertypeid
    WHERE c.transcriptid = {tid}
""")
print(speakers.to_string(index=False))

# ============================================================
# 3) Find Tim Cook and pull all his Presenter Speech + Answer segments
# ============================================================
cook = speakers[speakers['transcriptpersonname'].str.contains('Cook', case=False, na=False)]
print(f"\nMatching speakers for 'Cook':")
print(cook.to_string(index=False))

if len(cook) == 0:
    print("ERROR: Tim Cook not found on this call.")
else:
    cook_pid = int(cook['transcriptpersonid'].iloc[0])
    print(f"\nTim Cook transcriptpersonid = {cook_pid}")

    cook_text = safe_sql(f"""
        SELECT componentorder, transcriptcomponenttypeid, componenttext
        FROM ciq_transcripts.ciqtranscriptcomponent
        WHERE transcriptid = {tid}
          AND transcriptpersonid = {cook_pid}
          AND transcriptcomponenttypeid IN (2, 4)
        ORDER BY componentorder
    """)
    print(f"\nCook's segments on this call: {len(cook_text)}")
    print(f"  Presenter Speech (type 2): {(cook_text['transcriptcomponenttypeid']==2).sum()}")
    print(f"  Answer (type 4):          {(cook_text['transcriptcomponenttypeid']==4).sum()}")

    full = "\n\n".join(cook_text['componenttext'].astype(str).tolist())
    word_count = len(full.split())
    print(f"\nTotal Tim Cook speech: {len(full):,} chars, {word_count:,} words")
    print(f"Meets the 1,500-word threshold? {'YES' if word_count >= 1500 else 'NO'}")

    print(f"\nFirst 500 chars of concatenated Cook speech:")
    print(full[:500])
    print(f"\n...last 500 chars:")
    print(full[-500:])

db.close()