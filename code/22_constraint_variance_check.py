"""
Read-only check: how much of the Constrained (fin_constrained) dummy's variance
is within-firm vs. between-firm, i.e. how much would firm fixed effects absorb it.
Also reports switcher shares, and the same decomposition for size (log assets)
and firm age, since the thesis attributes the dummy's stickiness to those two
slow-moving drivers of the SA index.

Does not modify any existing script, panel, or output file.
"""

import pandas as pd

PANEL_PATH = "data/processed_extended/regression_panel.csv"
OUT_PATH = "output/constraint_variance_decomp.txt"

FIRM_COL = "gvkey"
YEAR_COL = "fyear"


def variance_decomp(df, firm_col, var_col):
    """Within-firm / between-firm variance shares for var_col, using the
    standard within/between decomposition (population variance, ddof=0)."""
    sub = df[[firm_col, var_col]].dropna()
    firm_means = sub.groupby(firm_col)[var_col].transform("mean")
    within = sub[var_col] - firm_means

    total_var = sub[var_col].var(ddof=0)
    within_var = within.var(ddof=0)
    between_var = firm_means.var(ddof=0)

    within_share = within_var / total_var if total_var > 0 else float("nan")
    between_share = between_var / total_var if total_var > 0 else float("nan")

    return {
        "total_var": total_var,
        "within_var": within_var,
        "between_var": between_var,
        "within_share": within_share,
        "between_share": between_share,
        "n_obs": len(sub),
    }


def switcher_share(df, firm_col, var_col):
    """Share of firms whose var_col value ever changes across the panel."""
    sub = df[[firm_col, var_col]].dropna()
    nunique = sub.groupby(firm_col)[var_col].nunique()
    n_firms = len(nunique)
    n_switchers = int((nunique > 1).sum())
    return n_switchers, n_firms, n_switchers / n_firms if n_firms > 0 else float("nan")


def main():
    df = pd.read_csv(PANEL_PATH)

    print("Loaded panel:", PANEL_PATH)
    print("Columns:", df.columns.tolist())
    print()
    print(df.head())
    print()

    lines = []
    lines.append("=" * 78)
    lines.append("VARIANCE DECOMPOSITION OF THE FINANCIAL-CONSTRAINT DUMMY (fin_constrained)")
    lines.append("=" * 78)
    lines.append(f"Source panel: {PANEL_PATH}")
    lines.append("")

    targets = [
        ("fin_constrained", "Constrained dummy (top-tercile SA index)"),
        ("size", "Firm size (log assets)"),
        ("age", "Firm age"),
    ]

    for col, label in targets:
        if col not in df.columns:
            lines.append(f"[skipped] column '{col}' not found in panel")
            lines.append("")
            continue

        dec = variance_decomp(df, FIRM_COL, col)
        n_sw, n_firms, sw_share = switcher_share(df, FIRM_COL, col)

        lines.append("-" * 78)
        lines.append(label + f"  [{col}]")
        lines.append("-" * 78)
        lines.append(f"N observations (non-missing):                 {dec['n_obs']}")
        lines.append(f"Total variance:                                {dec['total_var']:.6f}")
        lines.append(f"Between-firm variance:                         {dec['between_var']:.6f}")
        lines.append(f"Within-firm variance:                          {dec['within_var']:.6f}")
        lines.append(f"Between-firm share of total variance:          {dec['between_share']*100:.1f}%")
        lines.append(f"Within-firm share of total variance:           {dec['within_share']*100:.1f}%")
        lines.append("")
        lines.append(f"Firms with at least one value change (switchers): {n_sw} / {n_firms} ({sw_share*100:.1f}%)")
        lines.append(f"Never-switcher firms:                              {n_firms - n_sw} / {n_firms} ({(1-sw_share)*100:.1f}%)")
        lines.append("")

    lines.append("=" * 78)
    lines.append("INTERPRETATION")
    lines.append("=" * 78)
    lines.append(
        "A high between-firm variance share (and a low switcher share) means most of"
    )
    lines.append(
        "the variable's variation is cross-sectional, not within-firm over time --"
    )
    lines.append(
        "i.e. firm fixed effects absorb most of its identifying variation, leaving"
    )
    lines.append("little within-firm signal for the FE estimator to use.")
    lines.append("")

    out_text = "\n".join(lines)

    print(out_text)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(out_text + "\n")

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
