# cr_clean_windows.py
# Streamed, fast, low-RAM Clash Royale dataset cleaner for Random Forest modeling.
# Windows-friendly and optimized for battles.csv.

import os, time
import numpy as np
import pandas as pd

INPUT = r"C:\Users\Tyler Bacong\Desktop\battles.csv"  # <--- UPDATE THIS IF NEEDED
TEMP  = r"C:\Users\Tyler Bacong\Desktop\cr_rf_temp_players.csv"
OUT   = r"C:\Users\Tyler Bacong\Desktop\cr_rf_sample_100k.csv"

CHUNK = 150_000          # tune for RAM (150k–250k is ideal)
SAMPLE_RATE = 0.02       # 2% subsample for fast IQR
N_TARGET = 100_000
W, L = "winner", "loser"

USECOLS = [
    "arena.id", "average.startingTrophies",
    f"{W}.startingTrophies", f"{W}.trophyChange", f"{W}.crowns", f"{W}.elixir.average",
    f"{W}.troop.count", f"{W}.structure.count", f"{W}.spell.count",
    f"{W}.common.count", f"{W}.rare.count", f"{W}.epic.count", f"{W}.legendary.count",
    *[f"{W}.card{i}.id" for i in range(1, 9)],
    f"{L}.startingTrophies", f"{L}.trophyChange", f"{L}.crowns", f"{L}.elixir.average",
    f"{L}.troop.count", f"{L}.structure.count", f"{L}.spell.count",
    f"{L}.common.count", f"{L}.rare.count", f"{L}.epic.count", f"{L}.legendary.count",
    *[f"{L}.card{i}.id" for i in range(1, 9)],
]

DTYPES = {
    "arena.id": "Int32",
    "average.startingTrophies": "float64",
    **{f"{W}.card{i}.id": "Int32" for i in range(1, 9)},
    **{f"{L}.card{i}.id": "Int32" for i in range(1, 9)},
}

def reader():
    return pd.read_csv(
        INPUT,
        usecols=lambda c: c in USECOLS,
        dtype=DTYPES,
        chunksize=CHUNK,
        engine="c",
        on_bad_lines="skip",
        low_memory=True,
        memory_map=True
    )

# -------------------------------------------
#   PASS 1 — FAST IQR ESTIMATE
# -------------------------------------------
def fast_iqr():
    print("[1/3] Estimating IQR (subsampled, fast)")
    t0 = time.time()
    sample_vals = []
    scanned = 0

    for i, chunk in enumerate(reader(), start=1):
        col = pd.to_numeric(chunk["average.startingTrophies"], errors="coerce").dropna().to_numpy()
        if len(col):
            k = max(1, int(len(col) * SAMPLE_RATE))
            idx = np.random.choice(len(col), size=k, replace=False)
            sample_vals.append(col[idx])

        scanned += len(chunk)
        if i % 5 == 0:
            print(f"    scanned {scanned:_} rows…")

    if not sample_vals:
        print("    WARNING: no trophy data found.")
        return -np.inf, np.inf

    allv = np.concatenate(sample_vals)
    q1, q3 = np.percentile(allv, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(f"    IQR bounds: {lo:.2f} to {hi:.2f} [{time.time() - t0:.1f}s]")
    return lo, hi

# -------------------------------------------
#   PLAYER ROW CONVERSION
# -------------------------------------------
def players_from_chunk(b):
    w = pd.DataFrame({
        "arena_id": b["arena.id"],
        "avg_start_trophies": b["average.startingTrophies"],
        "label": 1,
        "player_start_trophies": b[f"{W}.startingTrophies"],
        "player_trophy_change": b[f"{W}.trophyChange"],
        "player_crowns": b[f"{W}.crowns"],
        "elixir_avg": b[f"{W}.elixir.average"],
        "troop_cnt": b[f"{W}.troop.count"],
        "structure_cnt": b[f"{W}.structure.count"],
        "spell_cnt": b[f"{W}.spell.count"],
        "common_cnt": b[f"{W}.common.count"],
        "rare_cnt": b[f"{W}.rare.count"],
        "epic_cnt": b[f"{W}.epic.count"],
        "legendary_cnt": b[f"{W}.legendary.count"],
    })
    w_cards = b[[f"{W}.card{i}.id" for i in range(1, 9)]].astype("Int64")
    w["cards_str"] = w_cards.fillna(-1).astype(str).agg("|".join, axis=1)

    l = pd.DataFrame({
        "arena_id": b["arena.id"],
        "avg_start_trophies": b["average.startingTrophies"],
        "label": 0,
        "player_start_trophies": b[f"{L}.startingTrophies"],
        "player_trophy_change": b[f"{L}.trophyChange"],
        "player_crowns": b[f"{L}.crowns"],
        "elixir_avg": b[f"{L}.elixir.average"],
        "troop_cnt": b[f"{L}.troop.count"],
        "structure_cnt": b[f"{L}.structure.count"],
        "spell_cnt": b[f"{L}.spell.count"],
        "common_cnt": b[f"{L}.common.count"],
        "rare_cnt": b[f"{L}.rare.count"],
        "epic_cnt": b[f"{L}.epic.count"],
        "legendary_cnt": b[f"{L}.legendary.count"],
    })
    l_cards = b[[f"{L}.card{i}.id" for i in range(1, 9)]].astype("Int64")
    l["cards_str"] = l_cards.fillna(-1).astype(str).agg("|".join, axis=1)

    return pd.concat([w, l], ignore_index=True)

# -------------------------------------------
#   PASS 2 — STREAM FILTER + PLAYER ROWS
# -------------------------------------------
def pass2_write(lo, hi):
    print("[2/3] Filtering to IQR + writing player rows")
    if os.path.exists(TEMP):
        os.remove(TEMP)

    t0 = time.time()
    header_written = False
    players_written = 0
    raw_seen = 0

    for i, chunk in enumerate(reader(), start=1):
        trophy = pd.to_numeric(chunk["average.startingTrophies"], errors="coerce")
        mask = trophy.between(lo, hi, inclusive="both")
        filt = chunk.loc[mask]

        if not filt.empty:
            players = players_from_chunk(filt)
            players.to_csv(TEMP, mode="a", index=False, header=not header_written)
            header_written = True
            players_written += len(players)

        raw_seen += len(chunk)
        if i % 5 == 0:
            elapsed = time.time() - t0
            print(f"    chunks={i}, raw={raw_seen:_}, players={players_written:_}, {raw_seen/max(elapsed,1):,.0f} rows/s")

    print(f"    TEMP written: {players_written:_} rows [{time.time()-t0:.1f}s]")
    return players_written

# -------------------------------------------
#   PASS 3 — STREAM SAMPLE TO 100k
# -------------------------------------------
def pass3_sample(total_players):
    print("[3/3] Sampling ~100k final rows")
    t0 = time.time()

    if total_players <= N_TARGET:
        df = pd.read_csv(TEMP)
        df.to_csv(OUT, index=False)
        print(f"    <=100k rows, copied directly → {OUT}")
        return

    keep_prob = N_TARGET / total_players
    rng = np.random.default_rng(42)

    parts = []
    kept = 0
    seen = 0

    for i, chunk in enumerate(pd.read_csv(TEMP, chunksize=CHUNK), start=1):
        mask = rng.random(len(chunk)) < keep_prob
        picked = chunk.loc[mask]
        if len(picked):
            parts.append(picked)
            kept += len(picked)

        seen += len(chunk)
        if i % 10 == 0:
            print(f"    chunks={i}, seen={seen:_}, kept={kept:_}")

    sample = pd.concat(parts, ignore_index=True)

    if len(sample) > N_TARGET:
        sample = sample.sample(n=N_TARGET, random_state=42)

    sample["trophy_bin"] = pd.cut(
        sample["avg_start_trophies"], bins=8, labels=[f"bin_{i}" for i in range(8)]
    )

    cols = [
        "label","arena_id","avg_start_trophies","player_start_trophies",
        "player_trophy_change","player_crowns","elixir_avg",
        "troop_cnt","structure_cnt","spell_cnt",
        "common_cnt","rare_cnt","epic_cnt","legendary_cnt",
        "trophy_bin","cards_str"
    ]
    sample[cols].to_csv(OUT, index=False)
    print(f"    wrote final → {OUT} [{time.time()-t0:.1f}s]")

# -------------------------------------------
# MAIN
# -------------------------------------------
def main():
    lo, hi = fast_iqr()
    total_players = pass2_write(lo, hi)
    pass3_sample(total_players)
    print("Done.")

if __name__ == "__main__":
    main()
