import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ============================================================
# 1) Load the LM dictionary and build word-sets
# ============================================================
lm = pd.read_csv('data/raw/lm_dictionary.csv')

# The flags are non-zero when the word belongs to that category.
positive_set    = set(lm.loc[lm['Positive']     != 0, 'Word'].str.upper())
negative_set    = set(lm.loc[lm['Negative']     != 0, 'Word'].str.upper())
uncertainty_set = set(lm.loc[lm['Uncertainty']  != 0, 'Word'].str.upper())
strong_modal    = set(lm.loc[lm['Strong_Modal'] != 0, 'Word'].str.upper())
weak_modal      = set(lm.loc[lm['Weak_Modal']   != 0, 'Word'].str.upper())
hedging_set     = uncertainty_set | weak_modal    # for inverse hedging

print("Loughran-McDonald word-set sizes:")
print(f"  Positive       : {len(positive_set):,}")
print(f"  Negative       : {len(negative_set):,}")
print(f"  Uncertainty    : {len(uncertainty_set):,}")
print(f"  Strong-modal   : {len(strong_modal):,}")
print(f"  Weak-modal     : {len(weak_modal):,}")
print(f"  Hedging (U+W)  : {len(hedging_set):,}")

# ============================================================
# 2) Tokeniser: uppercase, strip non-letters, split on whitespace
# ============================================================
TOKEN_RE = re.compile(r"[A-Z][A-Z'-]*")

def tokenise(text):
    return TOKEN_RE.findall(text.upper())

# ============================================================
# 3) Score every transcript in the final corpus
# ============================================================
final = pd.read_csv('data/processed/ceo_speech_final.csv',
                    dtype={'gvkey': str})
print(f"\nTranscripts to score: {len(final)}")

speech_dir = Path('data/processed/ceo_speech')

scores = []
for _, row in final.iterrows():
    fpath = speech_dir / row['filename']
    if not fpath.exists():
        continue
    text = fpath.read_text(encoding='utf-8')
    tokens = tokenise(text)
    n = len(tokens)
    if n == 0:
        continue

    n_pos     = sum(1 for t in tokens if t in positive_set)
    n_neg     = sum(1 for t in tokens if t in negative_set)
    n_strong  = sum(1 for t in tokens if t in strong_modal)
    n_hedge   = sum(1 for t in tokens if t in hedging_set)

    # Per-1,000-word scaling
    per_k = 1000.0 / n
    net_tone        = (n_pos - n_neg) * per_k
    strong_modality = n_strong         * per_k
    inverse_hedging = -n_hedge         * per_k    # negative => higher = more confident

    scores.append({
        'gvkey': row['gvkey'],
        'call_year': int(row['call_year']),
        'call_quarter': int(row['call_quarter']),
        'transcriptid': row['transcriptid'],
        'n_tokens': n,
        'net_tone': net_tone,
        'strong_modality': strong_modality,
        'inverse_hedging': inverse_hedging,
    })

call_scores = pd.DataFrame(scores)
call_scores.to_csv('data/processed/call_scores.csv', index=False)
print(f"Quarterly call scores computed: {len(call_scores)}")

# ============================================================
# 4) Aggregate to firm-year (within-year mean of quarterly scores)
# ============================================================
fy = (
    call_scores
    .groupby(['gvkey', 'call_year'])[['net_tone', 'strong_modality', 'inverse_hedging']]
    .mean()
    .reset_index()
    .rename(columns={'call_year': 'fyear'})
)
fy['n_calls_used'] = (
    call_scores.groupby(['gvkey', 'call_year']).size().values
)
print(f"\nFirm-year observations: {len(fy)}")

# ============================================================
# 5) Z-standardise each component across the panel
# ============================================================
for comp in ['net_tone', 'strong_modality', 'inverse_hedging']:
    z = (fy[comp] - fy[comp].mean()) / fy[comp].std(ddof=0)
    fy[f'z_{comp}'] = z

# Simple-average overconfidence index
fy['overconfidence'] = fy[['z_net_tone', 'z_strong_modality', 'z_inverse_hedging']].mean(axis=1)
# Two-component composite (net tone + inverse hedging only)
# Pre-registered robustness following Jonathan's supervisor feedback (June 2026)
# Motivated by the negative cross-correlation of strong modality with the other two components
fy['overconfidence_2comp'] = fy[['z_net_tone', 'z_inverse_hedging']].mean(axis=1)

# ============================================================
# 6) PCA aggregation (robustness alternative)
# ============================================================
X = fy[['z_net_tone', 'z_strong_modality', 'z_inverse_hedging']].values
pca = PCA(n_components=1)
fy['overconfidence_pca'] = pca.fit_transform(X)[:, 0]

# Flip sign if PCA loading on net_tone is negative (so higher = more confident)
if pca.components_[0, 0] < 0:
    fy['overconfidence_pca'] *= -1
    pca_loadings = -pca.components_[0]
else:
    pca_loadings = pca.components_[0]

print(f"\nPCA explained variance ratio (PC1): {pca.explained_variance_ratio_[0]:.3f}")
print(f"PCA loadings on [net_tone, strong_modality, inverse_hedging]: "
      f"{pca_loadings.round(3).tolist()}")

# ============================================================
# 7) Sanity checks
# ============================================================
print(f"\nDescriptive stats of overconfidence index:")
print(fy[['overconfidence', 'overconfidence_pca']].describe().to_string())

print(f"\nCorrelations between the three z-standardised components:")
print(fy[['z_net_tone', 'z_strong_modality', 'z_inverse_hedging']].corr().round(3).to_string())

print(f"\nCorrelation between simple-average index and PCA index:")
print(fy[['overconfidence', 'overconfidence_pca']].corr().iloc[0, 1].round(3))

# ============================================================
# 8) Save outputs + diagnostic plots
# ============================================================
fy.to_csv('data/processed/overconfidence_index.csv', index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, title in zip(axes,
                          ['z_net_tone', 'z_strong_modality', 'z_inverse_hedging'],
                          ['Net tone (z)', 'Strong modality (z)', 'Inverse hedging (z)']):
    ax.hist(fy[col].dropna(), bins=30, color='steelblue', edgecolor='white')
    ax.set_title(title)
    ax.set_xlabel('z-score')
plt.tight_layout()
plt.savefig('output/components_histograms.png', dpi=120)
plt.close()

plt.figure(figsize=(7, 5))
plt.hist(fy['overconfidence'].dropna(), bins=30, color='darkorange', edgecolor='white')
plt.title('Baseline Overconfidence Index (firm-year)')
plt.xlabel('Index value')
plt.ylabel('Firm-years')
plt.savefig('output/overconfidence_distribution.png', dpi=120)
plt.close()

print("\nSaved:")
print("  data/processed/call_scores.csv             <- quarterly raw scores")
print("  data/processed/overconfidence_index.csv    <- final firm-year index")
print("  output/components_histograms.png")
print("  output/overconfidence_distribution.png")