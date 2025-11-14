"""
Clash Royale – Deck Synergy & Win-Condition Strength (Random Forest)

Files needed in the same folder:
- cr_rf_sample_100k.csv
- CardMasterListSeason18_12082020.csv  (has team.card1.id, team.card1.name)
- Wincons.csv  (columns: id,card_id,card_name)

What this script outputs:
1. Model accuracy using ONLY deck composition (no trophy-based leakage)
2. Top overall card features (by importance)
3. Top win-condition cards (by model importance)
4. Empirical winrate and sample size for each win-condition card
5. Top synergy pairs (card pairs with highest predicted win probability)
"""

import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------------
# 1. Load match data
# --------------------------------------------------------
df = pd.read_csv("cr_rf_sample_100k.csv")

# Only keep valid rows
required_cols = ["cards_str", "label", "player_start_trophies"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Expected column '{c}' not found in data")

df = df.dropna(subset=["cards_str", "label", "player_start_trophies"])
df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

# --------------------------------------------------------
# 2. Restrict to IQR trophy range (focus on typical players)
# --------------------------------------------------------
q1 = df["player_start_trophies"].quantile(0.25)
q3 = df["player_start_trophies"].quantile(0.75)
mask_iqr = (df["player_start_trophies"] >= q1) & (df["player_start_trophies"] <= q3)
df = df[mask_iqr].reset_index(drop=True)

print("Rows after IQR trophy filter:", len(df))
print("Trophy range kept (IQR):", int(q1), "to", int(q3))

# --------------------------------------------------------
# 3. Load card master mapping (ID -> Name)
#    Expected columns: team.card1.id, team.card1.name
# --------------------------------------------------------
card_master = pd.read_csv("CardMasterListSeason18_12082020.csv")
card_master.columns = [c.lower() for c in card_master.columns]

if "team.card1.id" not in card_master.columns or "team.card1.name" not in card_master.columns:
    raise ValueError("Expected columns 'team.card1.id' and 'team.card1.name' in CardMasterListSeason18_12082020.csv")

id_to_name = dict(
    zip(
        card_master["team.card1.id"].astype(str),
        card_master["team.card1.name"]
    )
)

# --------------------------------------------------------
# 4. Load wincon list (Wincons.csv: id,card_id,card_name)
# --------------------------------------------------------
wincon_df = pd.read_csv("Wincons.csv")
wincon_df.columns = [c.lower() for c in wincon_df.columns]

expected_wincon_cols = {"id", "card_id", "card_name"}
if not expected_wincon_cols.issubset(set(wincon_df.columns)):
    raise ValueError(f"Wincons.csv must contain columns {expected_wincon_cols}, got {wincon_df.columns}")

wincon_ids = set(wincon_df["card_id"].astype(str))
wincon_names = set(wincon_df["card_name"])

# --------------------------------------------------------
# 5. Multi-hot encode deck composition (card IDs)
# --------------------------------------------------------
# columns like "26000021","28000004",...
card_matrix_id = df["cards_str"].str.get_dummies(sep="|").astype("int8")

# Rename columns from ID -> readable card name
card_col_names = []
for c in card_matrix_id.columns:
    # c is like '26000021'
    card_col_names.append(id_to_name.get(str(c), f"card_{c}"))

card_matrix = card_matrix_id.copy()
card_matrix.columns = card_col_names

# --------------------------------------------------------
# 6. Add "has_wincon" feature (deck contains at least one wincon card)
# --------------------------------------------------------
def deck_has_wincon(deck_str: str) -> int:
    ids = deck_str.split("|")
    return int(any(card_id in wincon_ids for card_id in ids))

df["has_wincon"] = df["cards_str"].apply(deck_has_wincon)

# --------------------------------------------------------
# 7. Build feature matrix X and target y
#    Only deck composition + has_wincon (no trophies, no crowns, etc.)
# --------------------------------------------------------
X = card_matrix.copy()
X["has_wincon"] = df["has_wincon"]
y = df["label"]

# --------------------------------------------------------
# 8. Train / test split
# --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------------
# 9. Train Random Forest on deck-only features
# --------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nRandom Forest accuracy using ONLY deck composition: {acc:.3f}\n")

# --------------------------------------------------------
# 10. Feature importances
# --------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns)

# Separate card features vs "has_wincon"
card_cols = [c for c in X.columns if c != "has_wincon"]
card_importances = importances[card_cols]
top_cards = card_importances.sort_values(ascending=False).head(20)

print("\nTop 20 most impactful cards for winning (by model importance):")
print(top_cards)

print("\nImportance of 'has_wincon' feature:")
print(importances["has_wincon"])

# --------------------------------------------------------
# 11. Win-condition importance (by model)
# --------------------------------------------------------
wincon_importances = card_importances[card_importances.index.isin(wincon_names)]
wincon_importances = wincon_importances.sort_values(ascending=False)

print("\nWin-condition cards ranked by model importance:")
print(wincon_importances)

# --------------------------------------------------------
# 12. Empirical winrate for each win-condition
# --------------------------------------------------------
wincon_stats = []
for name in wincon_names:
    if name in card_matrix.columns:
        mask = card_matrix[name] == 1
        games = mask.sum()
        if games >= 50:  # require minimum sample size for stability
            winrate = y[mask].mean()
            wincon_stats.append((name, winrate, games))

if wincon_stats:
    wincon_stats_df = pd.DataFrame(wincon_stats, columns=["wincon_card", "winrate", "games"])
    wincon_stats_df = wincon_stats_df.sort_values(by="winrate", ascending=False)

    print("\nEmpirical winrates for win-condition cards (min 50 games):")
    print(wincon_stats_df.head(20))
else:
    print("\nNo win-condition stats computed (check that wincon names match card names).")

# --------------------------------------------------------
# 13. Synergy pairs from model (top card pairs)
# --------------------------------------------------------
print("\nComputing synergy pairs among top cards (this may take a moment)...")

synergy_scores = {}
topN_cards = top_cards.index[:30]  # limit to top 30 cards for speed

# Use a zero baseline deck as template
baseline = X_train.iloc[0:1].copy()
baseline.loc[:, :] = 0

for c1, c2 in combinations(topN_cards, 2):
    row = baseline.copy()
    row[c1] = 1
    row[c2] = 1
    score = rf.predict_proba(row)[0, 1]  # predicted win probability
    synergy_scores[(c1, c2)] = score

top_synergies = sorted(synergy_scores.items(), key=lambda x: x[1], reverse=True)[:20]

print("\nTop 20 card synergy pairs (highest predicted win probability for the pair):")
for (c1, c2), score in top_synergies:
    print(f"{c1} + {c2} -> predicted win chance {score:.3f}")

# --------------------------------------------------------
# 14. Summary for presentation
# --------------------------------------------------------
print("\nSummary for presentation:")
print(f"- Trained on {len(df)} matches between {int(q1)} and {int(q3)} trophies.")
print(f"- Model uses ONLY deck composition (cards + has_wincon) to predict wins.")
print(f"- Test accuracy: {acc:.3f}")
print("- Top cards list shows which cards the model relies on most to spot winning decks.")
print("- Win-condition importance + empirical winrates show which wincons actually perform best.")
print("- Synergy pairs show card combinations with the highest modeled win probability.")

import matplotlib.pyplot as plt

# 1) Top 15 cards by model importance
top15_cards = top_cards.head(15)

plt.figure(figsize=(8, 6))
top15_cards.sort_values().plot(kind="barh")
plt.xlabel("Random Forest feature importance")
plt.title("Top 15 cards associated with winning\n(IQR 4388 to 4912 trophies)")
plt.tight_layout()
plt.show()


# 2) Win condition winrates (top 10 by winrate, min 50 games already enforced)
if "wincon_stats_df" in globals() and not wincon_stats_df.empty:
    top_wincons = wincon_stats_df.sort_values("winrate", ascending=False).head(10)

    plt.figure(figsize=(8, 6))
    plt.bar(top_wincons["wincon_card"], top_wincons["winrate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Empirical winrate")
    plt.ylim(0.45, 0.55)  # adjust if you want a different zoom
    plt.title("Top 10 win condition cards by empirical winrate")
    plt.tight_layout()
    plt.show()
else:
    print("No wincon_stats_df available or it is empty. Skipping wincon winrate plot.")


# 3) Top 10 synergy pairs by predicted win probability
if "top_synergies" in globals() and len(top_synergies) > 0:
    synergy_rows = []
    for (c1, c2), score in top_synergies:
        pair_name = f"{c1} + {c2}"
        synergy_rows.append((pair_name, score))

    synergy_df = pd.DataFrame(synergy_rows, columns=["pair", "pred_win_prob"])
    top10_synergies = synergy_df.head(10)

    plt.figure(figsize=(8, 6))
    plt.barh(top10_synergies["pair"], top10_synergies["pred_win_prob"])
    plt.xlabel("Predicted win probability for the pair")
    plt.title("Top 10 card synergy pairs (model based)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
else:
    print("No synergy data available. Skipping synergy plot.")


# 4) Baseline comparison printout
if "y_test" in globals():
    win_rate = y_test.mean()
    majority_baseline = max(win_rate, 1 - win_rate)
    print("\nBaseline accuracy if we always predict the majority class:",
          round(majority_baseline, 3))
    print("Win rate in test set (always predict win):", round(win_rate, 3))
