import wrds
import pandas as pd

db = wrds.Connection(wrds_username='sidt28')

# Pull S&P 500 constituents as of 1 January 2018
sp500 = db.raw_sql("""
    SELECT permno, start, ending
    FROM crsp.msp500list
    WHERE start <= '2018-01-01' AND ending >= '2018-01-01'
""")

print(f"S&P 500 constituents on 2018-01-01: {len(sp500)}")
print(sp500.head())

# Save for later steps
sp500.to_csv('data/raw/sp500_2018.csv', index=False)
print("Saved to data/raw/sp500_2018.csv")

# ============================================================
# Step 3: Map PERMNO -> GVKEY using CRSP/Compustat link table
# ============================================================
permnos = tuple(sp500['permno'].astype(int).tolist())

link = db.raw_sql(f"""
    SELECT lpermno AS permno, gvkey, linkdt, linkenddt, linktype, linkprim
    FROM crsp.ccmxpf_linktable
    WHERE lpermno IN {permnos}
      AND linktype IN ('LU', 'LC')
      AND linkprim IN ('P', 'C')
      AND linkdt <= '2018-01-01'
      AND (linkenddt >= '2018-01-01' OR linkenddt IS NULL)
""")

print(f"\nPERMNO->GVKEY links found: {len(link)}")
universe = sp500.merge(link[['permno', 'gvkey']], on='permno', how='inner')
print(f"Universe with GVKEYs: {len(universe)}")

universe.to_csv('data/processed/universe_step1.csv', index=False)
print("Saved to data/processed/universe_step1.csv")

# ============================================================
# Step 4: Pull SIC codes and company names from Compustat
# ============================================================
gvkeys = tuple(universe['gvkey'].tolist())

sic_data = db.raw_sql(f"""
    SELECT gvkey, sic, conm
    FROM comp.company
    WHERE gvkey IN {gvkeys}
""")

universe = universe.merge(sic_data, on='gvkey', how='left')
universe['sic'] = pd.to_numeric(universe['sic'], errors='coerce').astype('Int64')

print(f"\nAfter adding SIC codes: {len(universe)}")
print(f"Missing SIC codes: {universe['sic'].isna().sum()}")
print(universe[['permno', 'gvkey', 'conm', 'sic']].head(10))

universe.to_csv('data/processed/universe_with_sic.csv', index=False)
print("Saved to data/processed/universe_with_sic.csv")
db.close()