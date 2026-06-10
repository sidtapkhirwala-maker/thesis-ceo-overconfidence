import pandas as pd
import re

# ============================================================
# Load the universe with SIC codes
# ============================================================
universe = pd.read_csv('data/processed/universe_with_sic.csv', dtype={'gvkey': str})
print(f"Starting universe: {len(universe)} firms")

# ============================================================
# Parse Siccodes12.txt into a SIC -> FF12 industry mapping
# ============================================================
ff12_ranges = []  # list of (industry_number, industry_name, sic_low, sic_high)

with open('data/raw/Siccodes12.txt', 'r') as f:
    current_industry = None
    current_name = None
    for raw_line in f:
        line = raw_line.strip()
        if not line:
            continue
        # Header line like "1 NoDur  Consumer NonDurables ..."
        header_match = re.match(r'^\s*(\d+)\s+(\S+)\s', raw_line)
        # SIC range line like "0100-0199"
        range_match = re.match(r'^\s*(\d{4})-(\d{4})', line)
        if range_match and current_industry is not None:
            low = int(range_match.group(1))
            high = int(range_match.group(2))
            ff12_ranges.append((current_industry, current_name, low, high))
        elif header_match:
            current_industry = int(header_match.group(1))
            current_name = header_match.group(2)

print(f"Loaded {len(ff12_ranges)} SIC ranges across FF12 industries")

# ============================================================
# Assign FF12 industry to each firm
# ============================================================
def sic_to_ff12(sic):
    if pd.isna(sic):
        return (12, 'Other')  # FF12 "Other" by default
    sic = int(sic)
    for ind_num, ind_name, low, high in ff12_ranges:
        if low <= sic <= high:
            return (ind_num, ind_name)
    return (12, 'Other')  # Anything unmatched falls into "Other"

universe[['ff12_num', 'ff12_name']] = universe['sic'].apply(
    lambda s: pd.Series(sic_to_ff12(s))
)

print("\nFF12 industry distribution BEFORE exclusions:")
print(universe['ff12_name'].value_counts())

# ============================================================
# Apply the two pre-registered exclusions
# FF12 industry 8 = Utilities, FF12 industry 11 = Finance
# ============================================================
before = len(universe)
universe = universe[~universe['ff12_num'].isin([8, 11])].copy()
print(f"\nDropped {before - len(universe)} firms (Utilities + Financials)")
print(f"Universe after exclusions: {len(universe)} firms")

print("\nFF12 industry distribution AFTER exclusions:")
print(universe['ff12_name'].value_counts())

universe.to_csv('data/processed/universe_after_exclusions.csv', index=False)
print("\nSaved to data/processed/universe_after_exclusions.csv")