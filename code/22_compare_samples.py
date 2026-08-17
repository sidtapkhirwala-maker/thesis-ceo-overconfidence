"""
Side-by-side comparison: primary 80-firm vs extended 327-firm.
Generates the exact table to drop into Appendix C of the thesis.
"""
import pandas as pd
import os

os.makedirs('output/extended/for_thesis', exist_ok=True)

# Primary panel
panel_primary = pd.read_csv('data/processed/regression_panel.csv',
                            dtype={'gvkey': str})
# Extended panel
panel_ext = pd.read_csv('data/processed_extended/regression_panel.csv',
                        dtype={'gvkey': str})

summary = pd.DataFrame({
    'Metric': ['Firms in sample',
               'Firm-years (H1 panel)',
               'Firm-years (H2, gross issuance)',
               'Firm-years (H2, strict net)',
               'Mean book leverage (t+1)',
               'Mean debt share (t+1, gross)',
               'Mean overconfidence (3-comp)',
               'SD overconfidence (3-comp)'],
    'Primary (80 firms)': [
        panel_primary['gvkey'].nunique(),
        panel_primary['book_leverage_t1'].notna().sum(),
        panel_primary['debt_share_t1'].notna().sum(),
        panel_primary['debt_share_strict_t1'].notna().sum() if 'debt_share_strict_t1' in panel_primary.columns else 'n/a',
        round(panel_primary['book_leverage_t1'].mean(), 3),
        round(panel_primary['debt_share_t1'].mean(), 3),
        round(panel_primary['overconfidence'].mean(), 4),
        round(panel_primary['overconfidence'].std(), 3),
    ],
    'Extended (327 firms)': [
        panel_ext['gvkey'].nunique(),
        panel_ext['book_leverage_t1'].notna().sum(),
        panel_ext['debt_share_t1'].notna().sum(),
        panel_ext['debt_share_strict_t1'].notna().sum() if 'debt_share_strict_t1' in panel_ext.columns else 'n/a',
        round(panel_ext['book_leverage_t1'].mean(), 3),
        round(panel_ext['debt_share_t1'].mean(), 3),
        round(panel_ext['overconfidence'].mean(), 4),
        round(panel_ext['overconfidence'].std(), 3),
    ],
})

print("=" * 70)
print("SAMPLE COMPARISON: Primary vs Extended")
print("=" * 70)
print(summary.to_string(index=False))
summary.to_csv('output/extended/for_thesis/sample_comparison.csv', index=False)
print("\nSaved: output/extended/for_thesis/sample_comparison.csv")