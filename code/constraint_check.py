import pandas as pd

# Load your existing H2 panel data (adjust path/filename to match your pipeline)
panel = pd.read_csv("data/processed_extended/h2_panel.csv")  # <-- use your actual file

firm_means = panel.groupby('firm_id')['constrained'].transform('mean')
within_var = (panel['constrained'] - firm_means).var()
total_var = panel['constrained'].var()

within_share = within_var / total_var
print(f"Within-firm share of variance in Constrained: {within_share:.1%}")

# Also report the more intuitive version
switchers = panel.groupby('firm_id')['constrained'].nunique()
pct_switchers = (switchers > 1).mean()
print(f"Share of firms that ever switch constraint status: {pct_switchers:.1%}")