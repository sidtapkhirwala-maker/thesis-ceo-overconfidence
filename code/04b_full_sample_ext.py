"""
Extended sample selector.
Uses the 327 firms that passed the 2018-2022 continuous-listing filter.
No random draw, no stratification - just the full eligible pool.
Frozen eligibility frame: firms eligible based on 2018-2022 data quality,
even if some delist or get acquired during the 2023-2024 forward extension.
"""
import pandas as pd

eligible = pd.read_csv('data/processed_extended/universe_eligible.csv',
                       dtype={'gvkey': str})
print(f"Eligible firms (frozen 2018-2022 frame): {len(eligible)}")
print("\nBy FF12 industry:")
print(eligible['ff12_name'].value_counts().to_string())

with open('data/processed_extended/sample_firms.csv', 'w') as f:
    f.write("# Extended sample: full 327-firm eligible pool\n")
    f.write("# Frame frozen on 2018-2022 continuous-listing filter\n")
    f.write("# Forward-extended to 2024 (delisted firms keep their 2018-2022 data)\n")
    f.write(f"# Sample size: {len(eligible)}\n")
    f.write("# Sensitivity analysis; primary remains the pre-registered 80-firm draw\n")
eligible.to_csv('data/processed_extended/sample_firms.csv',
                mode='a', index=False)
print(f"\nSaved: data/processed_extended/sample_firms.csv")