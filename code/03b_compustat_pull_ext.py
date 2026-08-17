"""
Compustat pull for the extended sample.
Window: fiscal 2017 through 2025 (2017 for lagged controls, 2025 for t+1
outcome of 2024 index). Firms = the 327 frozen-frame eligible pool.
"""
import wrds
import pandas as pd

sample = pd.read_csv('data/processed_extended/sample_firms.csv',
                     comment='#', dtype={'gvkey': str})
print(f"Sample firms (extended): {len(sample)}")
gvkeys = tuple(sample['gvkey'].tolist())

db = wrds.Connection(wrds_username='sidt28')

print("\nPulling Compustat funda 2017-2025...")
funda = db.raw_sql(f"""
    SELECT gvkey, datadate, fyear, at, dltt, dlc, oibdp, ppent, che,
           prcc_f, csho, dltis, dltr, sstk, prstkc, sich
    FROM comp.funda
    WHERE gvkey IN {gvkeys}
      AND fyear BETWEEN 2017 AND 2025
      AND indfmt='INDL' AND datafmt='STD'
      AND popsrc='D' AND consol='C'
""")
print(f"Raw Compustat rows pulled: {len(funda)}")
print(f"Distinct firms with data: {funda['gvkey'].nunique()}")
print(f"\nCoverage by fyear:")
print(funda.groupby('fyear').size().to_string())

# Quick check on 2025 coverage - if it's much sparser than 2024 we'll trim back
n_2024 = (funda['fyear'] == 2024).sum()
n_2025 = (funda['fyear'] == 2025).sum()
print(f"\n2024 firm-rows: {n_2024}")
print(f"2025 firm-rows: {n_2025}")
print(f"2025 coverage relative to 2024: {n_2025/n_2024*100:.1f}%" if n_2024 else "n/a")

funda.to_csv('data/raw/compustat/funda_raw_ext.csv', index=False)

# Apply the FROZEN eligibility filter from the primary pipeline
# i.e. firms must have had non-missing key vars in EVERY year 2018-2022
# but we keep ALL their years of data 2017-2025 for the extended panel.
# Note: this filter has already been applied to define the 327-firm sample.
# We just save the full Compustat panel for those firms.

funda.to_csv('data/processed_extended/compustat_panel.csv', index=False)
print(f"\nSaved: data/processed_extended/compustat_panel.csv ({len(funda)} rows)")

db.close()