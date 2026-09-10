from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\xl\OneDrive - Universitat Ramón Llull\git\practice")
PANEL_DIR = Path(r"C:\Users\xl\OneDrive - Universitat Ramón Llull\bot_paper\panel_construction")
RAW_ROBOTS = [
    Path(r"D:\anoconda\bot_paper_data\robots_panel_monthly.csv"),
    Path(r"D:\anoconda\bot_paper_data\robots_panel_monthly_new.csv"),
]
FIRM_PANEL = PANEL_DIR / "firm_panel_monthly.csv"
EVENT_PANEL = PANEL_DIR / "edgar_filing_event_panel.csv"

TREATMENT_OUT = ROOT / "bot_family_specific_treatments.csv"
EVENT_OUT = ROOT / "bot_family_event_estimation.csv"
DIAG_OUT = ROOT / "bot_family_treatment_diagnostics.csv"


BOT_FAMILIES = {
    # Strict AI discovery/search bots: official search/retrieval agents.
    "core_ai_search": {"OAI-SearchBot", "PerplexityBot", "Claude-SearchBot"},
    # Broader AI-adjacent discovery bots; kept separate because Applebot/PetalBot/Anomura
    # are less cleanly "LLM search" than the core group.
    "broad_ai_discovery": {
        "OAI-SearchBot",
        "PerplexityBot",
        "Claude-SearchBot",
        "Applebot",
        "PetalBot",
        "Anomura",
    },
    # Training/control crawlers. Google-Extended is not a crawler in the same way as GPTBot,
    # but it is a robots token controlling use for model training, so it belongs here.
    "training_llm_core": {"GPTBot", "ClaudeBot", "Google-Extended", "CCBot"},
    "traditional_search": {
        "Googlebot",
        "bingbot",
        "DuckDuckBot",
        "YandexBot",
        "Baiduspider",
        "Yeti",
        "SeznamBot",
    },
    "oai_search": {"OAI-SearchBot"},
    "gptbot": {"GPTBot"},
}


def ym_to_int(s: pd.Series) -> pd.Series:
    ss = s.astype("string").str.slice(0, 7)
    return pd.to_numeric(ss.str.replace("-", "", regex=False), errors="coerce").astype("Int64")


def norm_host(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower().str.rstrip(".")


def summarize_group(panel: pd.DataFrame, family: str) -> pd.DataFrame:
    valid = f"{family}_valid"
    block = f"{family}_block"
    df = panel[["gvkey", "ym_int", valid, block]].copy()
    df = df[df[valid].eq(1)]

    base = df.groupby("gvkey", observed=True)["ym_int"].min().rename(f"{family}_first_valid")
    clean = (
        df[df[block].eq(0)]
        .groupby("gvkey", observed=True)["ym_int"]
        .min()
        .rename(f"{family}_first_clean")
    )
    first = (
        df[df[block].eq(1)]
        .groupby("gvkey", observed=True)["ym_int"]
        .min()
        .rename(f"{family}_first_block")
    )

    out = pd.concat([base, clean, first], axis=1).reset_index()
    fb = f"{family}_first_block"
    fc = f"{family}_first_clean"
    fv = f"{family}_first_valid"
    out[f"{family}_has_valid"] = out[fv].notna().astype("int8")
    out[f"{family}_has_block"] = out[fb].notna().astype("int8")
    out[f"{family}_clean_before_block"] = (
        out[fb].notna() & out[fc].notna() & (out[fc] < out[fb])
    ).astype("int8")
    out[f"{family}_canon_treated"] = (
        out[f"{family}_has_block"].eq(1) & out[f"{family}_clean_before_block"].eq(1)
    ).astype("int8")
    out[f"{family}_left_censored"] = (
        out[f"{family}_has_block"].eq(1) & out[f"{family}_clean_before_block"].eq(0)
    ).astype("int8")
    out[f"{family}_clean_never"] = (
        out[f"{family}_has_valid"].eq(1) & out[f"{family}_has_block"].eq(0)
    ).astype("int8")
    return out


def read_firm_month_map() -> pd.DataFrame:
    usecols = ["gvkey", "host", "year_month"]
    fm = pd.read_csv(FIRM_PANEL, usecols=usecols, dtype={"gvkey": "string", "host": "string", "year_month": "string"})
    fm["host_key"] = norm_host(fm["host"])
    fm["ym_int"] = ym_to_int(fm["year_month"])
    fm = fm.dropna(subset=["host_key", "ym_int"])
    fm["ym_int"] = fm["ym_int"].astype("int32")
    fm = fm[["gvkey", "host_key", "year_month", "ym_int"]].drop_duplicates()
    return fm


def build_treatments() -> pd.DataFrame:
    start = time.time()
    fm = read_firm_month_map()
    print(f"[map] firm-host-month rows: {len(fm):,}; firms: {fm['gvkey'].nunique():,}; hosts: {fm['host_key'].nunique():,}", flush=True)

    all_bots = set().union(*BOT_FAMILIES.values())
    bot_to_families: dict[str, list[str]] = {}
    for family, bots in BOT_FAMILIES.items():
        for bot in bots:
            bot_to_families.setdefault(bot, []).append(family)

    collapsed_chunks: list[pd.DataFrame] = []
    usecols = ["host", "year_month", "bot", "blocked_any", "specific_rule"]

    for raw in RAW_ROBOTS:
        file_start = time.time()
        print(f"[raw] scanning {raw}", flush=True)
        for i, chunk in enumerate(pd.read_csv(raw, usecols=usecols, dtype={"host": "string", "year_month": "string", "bot": "string"}, chunksize=1_000_000), 1):
            chunk = chunk[chunk["bot"].isin(all_bots)].copy()
            if chunk.empty:
                continue
            chunk["host_key"] = norm_host(chunk["host"])
            merged = chunk.merge(fm, on=["host_key", "year_month"], how="inner")
            if merged.empty:
                continue
            merged["blocked_any_n"] = pd.to_numeric(merged["blocked_any"], errors="coerce")
            merged["specific_rule_n"] = pd.to_numeric(merged["specific_rule"], errors="coerce")

            pieces: list[pd.DataFrame] = []
            for bot, families in bot_to_families.items():
                bot_rows = merged[merged["bot"].eq(bot)]
                if bot_rows.empty:
                    continue
                for family in families:
                    tmp = bot_rows[["gvkey", "ym_int", "blocked_any_n", "specific_rule_n"]].copy()
                    tmp[f"{family}_valid"] = tmp["blocked_any_n"].notna().astype("int8")
                    tmp[f"{family}_block"] = (
                        tmp["blocked_any_n"].eq(1) & tmp["specific_rule_n"].eq(1)
                    ).astype("int8")
                    pieces.append(tmp[["gvkey", "ym_int", f"{family}_valid", f"{family}_block"]])

            if pieces:
                mini = pd.concat(pieces, ignore_index=True)
                mini = mini.groupby(["gvkey", "ym_int"], as_index=False, observed=True).max(numeric_only=True)
                collapsed_chunks.append(mini)

            if i % 10 == 0:
                print(f"  chunk {i:>3}; elapsed {time.time() - file_start:,.0f}s; collapsed pieces {len(collapsed_chunks):,}", flush=True)

    if not collapsed_chunks:
        raise RuntimeError("No matching bot rows were found in raw robots panels.")

    panel = pd.concat(collapsed_chunks, ignore_index=True)
    for family in BOT_FAMILIES:
        for suffix in ["valid", "block"]:
            col = f"{family}_{suffix}"
            if col not in panel.columns:
                panel[col] = 0
    panel = panel.groupby(["gvkey", "ym_int"], as_index=False, observed=True).max(numeric_only=True)
    print(f"[collapse] firm-month rows with selected bots: {len(panel):,}; firms: {panel['gvkey'].nunique():,}", flush=True)

    treatments: pd.DataFrame | None = None
    for family in BOT_FAMILIES:
        summ = summarize_group(panel, family)
        treatments = summ if treatments is None else treatments.merge(summ, on="gvkey", how="outer")

    assert treatments is not None
    treatments = treatments.sort_values("gvkey")
    treatments.to_csv(TREATMENT_OUT, index=False)
    print(f"[write] {TREATMENT_OUT} ({len(treatments):,} firms), elapsed {time.time() - start:,.0f}s", flush=True)
    return treatments


def write_diagnostics(treatments: pd.DataFrame) -> None:
    rows = []
    for family in BOT_FAMILIES:
        rows.append(
            {
                "family": family,
                "bots": ";".join(sorted(BOT_FAMILIES[family])),
                "has_valid": int(treatments[f"{family}_has_valid"].fillna(0).sum()),
                "canon_treated": int(treatments[f"{family}_canon_treated"].fillna(0).sum()),
                "left_censored": int(treatments[f"{family}_left_censored"].fillna(0).sum()),
                "clean_never": int(treatments[f"{family}_clean_never"].fillna(0).sum()),
                "first_nonmissing": int(treatments[f"{family}_first_block"].notna().sum()),
            }
        )
    pd.DataFrame(rows).to_csv(DIAG_OUT, index=False)
    print(f"[write] {DIAG_OUT}", flush=True)


def build_event_estimation(treatments: pd.DataFrame) -> None:
    usecols = ["gvkey", "filing_ym_int", "filing_ym_n", "in_sample", "abn_amihud_mean_0_5"]
    ev = pd.read_csv(EVENT_PANEL, usecols=usecols, dtype={"gvkey": "string"})
    ev = ev[ev["in_sample"].eq(1)].copy()
    ev = ev.dropna(subset=["gvkey", "filing_ym_int", "filing_ym_n", "abn_amihud_mean_0_5"])
    ev["filing_ym_int"] = pd.to_numeric(ev["filing_ym_int"], errors="coerce").astype("Int64")
    ev["filing_ym_n"] = pd.to_numeric(ev["filing_ym_n"], errors="coerce").astype("Int64")
    ev = ev.dropna(subset=["filing_ym_int", "filing_ym_n"])

    keep_cols = ["gvkey"]
    for family in BOT_FAMILIES:
        keep_cols.extend(
            [
                f"{family}_first_block",
                f"{family}_canon_treated",
                f"{family}_left_censored",
                f"{family}_clean_never",
            ]
        )
    ev = ev.merge(treatments[keep_cols], on="gvkey", how="left")

    for family in BOT_FAMILIES:
        for suffix in ["canon_treated", "left_censored", "clean_never"]:
            col = f"{family}_{suffix}"
            ev[col] = ev[col].fillna(0).astype("int8")

    ev.to_csv(EVENT_OUT, index=False)
    print(f"[write] {EVENT_OUT} ({len(ev):,} filing events; {ev['gvkey'].nunique():,} firms)", flush=True)


def main() -> None:
    treatments = build_treatments()
    write_diagnostics(treatments)
    build_event_estimation(treatments)


if __name__ == "__main__":
    main()
