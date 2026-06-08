import pandas as pd

still = pd.read_csv('data/processed/ceo_speech_still_unmatched.csv',
                    dtype={'gvkey': str})
print(f"Still unmatched: {len(still)}")

# Reasons breakdown
print("\nReason breakdown:")
print(still['rescue_reason'].value_counts().to_string())

# Show one representative example from each of the top 5 firms
print("\n=== One example per top-5 firm ===")
top_firms = still.groupby('gvkey').size().sort_values(ascending=False).head(5).index
for gvkey in top_firms:
    sub = still[still['gvkey'] == gvkey].iloc[0]
    print(f"\nGVKEY {gvkey} ({sub.get('conm','?')}) - "
          f"{sub.get('call_year','?')} Q{sub.get('call_quarter','?')}:")
    print(f"  CEO candidates : {sub.get('ceo_candidates','')}")
    print(f"  Best speaker   : {sub.get('best_speaker','')}")
    print(f"  Rescue reason  : {sub.get('rescue_reason','')}")