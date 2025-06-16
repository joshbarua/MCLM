import re
from pathlib import Path
from itertools import cycle
from typing import List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

###############################################################################
# Generic helper utilities                                                   #
###############################################################################

def _find_token(string: str, tokens: Sequence[str]) -> str | None:
    """Return the *first* token (case‑insensitive) that occurs in *string*."""
    low = string.lower()
    for tok in tokens:
        if tok.lower() in low:
            return tok
    return None


def _step_number(model: str) -> int:
    m = re.search(r"checkpoint-(\d+)", str(model))
    return int(m.group(1)) if m else 0

###############################################################################
# Main plotting routine                                                      #
###############################################################################

def plot_by_tokens(
    csv_paths: List[str],
    group_tokens: Sequence[str],   # determines line‑style groupings
    series_tokens: Sequence[str],  # determines colour groupings & column names
    y_label: str = "Metric (%)",
    out_file: str = "plot.png",
):
    """Plot curves where:
        • **Line style** encodes which *group_token* is found in the *model* path.
        • **Colour** encodes which *series_token* is being plotted (expects a col
          with that exact *title‑case* name, e.g. "english" → "English").

    The function keeps just one CSV per (group_token, series_token) pair:  the
    file that provides the most non‑null cells for that series, so sparse files
    never override dense ones.
    """

    # ── ingest all CSVs ────────────────────────────────────────────────────
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df["source"] = Path(p).name
        dfs.append(df)
    big = pd.concat(dfs, ignore_index=True, sort=False)

    # ── identify *group* for every row (token from group_tokens) ──────────
    big["group"] = big["model"].apply(lambda s: _find_token(str(s), group_tokens))

    # rows that don’t match any token are labelled "Other"
    big["group"].fillna("Other", inplace=True)

    # ── map series tokens to actual DataFrame column names (title‑case) ───
    #     e.g. "english" → "English".  Caller is responsible the column exists.
    series_cols = {tok: tok.title() for tok in series_tokens}

    # ensure requested columns exist (otherwise Pandas will raise later) ----
    missing = [c for c in series_cols.values() if c not in big.columns]
    if missing:
        raise ValueError(f"Missing columns in CSVs: {missing}")

    # ── select best CSV per (group, series) pair ---------------------------
    best_src: dict[tuple[str, str], str] = {}
    best_count: dict[tuple[str, str], int] = {}

    for src, block in big.groupby("source"):
        for grp, gblock in block.groupby("group"):
            for tok, col in series_cols.items():
                nn = gblock[col].notna().sum()
                if nn == 0:
                    continue
                key = (grp, tok)
                if key not in best_count or nn > best_count[key]:
                    best_count[key] = nn
                    best_src[key] = src

    # ── zero‑out inferior rows -------------------------------------------
    for idx, row in big.iterrows():
        grp, src = row["group"], row["source"]
        for tok, col in series_cols.items():
            key = (grp, tok)
            if key in best_src and src != best_src[key]:
                big.at[idx, col] = pd.NA

    # ── numeric cast + step extraction ------------------------------------
    for col in series_cols.values():
        big[col] = pd.to_numeric(big[col], errors="coerce")
    big["steps"] = big["model"].apply(_step_number)

    # ── styling maps -------------------------------------------------------
    colours = plt.get_cmap("tab10")
    colour_map = {tok: colours(i) for i, tok in enumerate(series_tokens)}
    linestyles = dict(zip(sorted(big["group"].unique()), cycle(["-", "--", ":", "-."])))

    # ── plotting -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    for grp in sorted(big["group"].unique()):
        grp_rows = big[big["group"] == grp]
        for tok in series_tokens:
            col = series_cols[tok]
            sub = grp_rows.dropna(subset=[col]).sort_values("steps")
            if sub.empty:
                continue
            ax.plot(
                sub["steps"],
                sub[col],
                marker="o",
                color=colour_map[tok],
                linestyle=linestyles[grp],
                linewidth=2,
            )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # ── legends outside plot area ----------------------------------------
    series_handles = [
        plt.Line2D([0], [0], color=colour_map[tok], lw=3) for tok in series_tokens
    ]
    group_handles = [
        plt.Line2D([0], [0], color="black", lw=2, linestyle=linestyles[grp])
        for grp in linestyles
    ]

    first = ax.legend(
        series_handles,
        series_tokens,
        title="Series (colour)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.57),
        frameon=False,
    )
    ax.add_artist(first)

    ax.legend(
        group_handles,
        linestyles.keys(),
        title="Group (line style)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.12),
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()

###############################################################################
# Example usage                                                               #
###############################################################################
if __name__ == "__main__":
    plot_by_tokens(
        csv_paths=["score_result/math500.csv", "score_result/math100.csv"],
        group_tokens=["Qwen2.5-7B-base", "Qwen3-8B-base"],
        series_tokens=["english", "swahili", "japanese", "french", "latvian"],
        y_label="MATH Accuracy (%)",
        out_file="qwen_accuracy_vs_steps.png",
    )