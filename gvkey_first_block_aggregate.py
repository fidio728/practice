"""gvkey_first_block_aggregate.py — collapse daily firmhostbot events
to gvkey-level adoption variables and a monthly adoption curve.

Why aggregate at gvkey × bot_group, not gvkey × any-bot:
  Monthly main treatment is `search_sitemap_block` (search-group only),
  so daily timing should mirror that and not blend group semantics.
  Treatment-mechanism analysis uses search; training/assistant/control
  are mechanism / placebo / sanity-check arms.

Per-bot timing rule:
  For each (gvkey, bot) pair, the gvkey's "first transition for this
  bot" = the EARLIEST first_observed_blocked_ts across all hosts that
  map to this gvkey. This is the conservative-late approach: as soon as
  ANY host of the firm flips to disallow for this bot, we record the
  firm as adopted.

Per-group timing rule:
  first_<group>_block_ts = min(first_observed_blocked_ts) across all
  bots in <group> for this gvkey, and interval_days_<group> is the
  interval_days of THAT specific (gvkey, bot, host) transition (i.e.,
  the timing-precision of the earliest event we recorded for the
  group). This couples adoption ts and its observability cleanly.

AI aggregate:
  first_any_ai_block_ts = min across (search ∪ training ∪ assistant);
  control is excluded so the variable is interpretable as "first time
  the firm blocked an AI-affiliated crawler."

Outputs (E:/bot_paper_data):

  gvkey_first_block_dates.csv      — one row per gvkey
    gvkey
    first_<group>_block_ts        (14-digit Wayback ts) for group in
    first_<group>_block_month      (YYYYMM) {search, training,
    interval_days_<group>            assistant, control}
    timing_quality_<group>           one of {le7, le14, le30, wide,
                                              ""}, "" if no event
    first_<group>_block_bot          (the bot that actually flipped)
    first_<group>_block_host         (which host triggered the flip)
    first_any_ai_block_ts            min over search+training+assistant
    first_any_ai_block_month
    first_any_ai_block_bot
    first_any_ai_block_host

  gvkey_adoption_curve_monthly.csv  — one row per calendar month from
    month_yyyymm                     min observed to max observed
    n_at_risk_gvkeys                 denominator (unique gvkeys in
                                     firmhostbot CSV, constant per row)
    <group>_new_count                gvkeys that adopted in this month
    <group>_cum_count                cumulative gvkeys adopted by EOM
    <group>_cum_share                cum_count / n_at_risk
                                     for group in {search, training,
                                                   assistant, control}
    <bot>_new_count                  same per-bot for the four headline
    <bot>_cum_count                  AI bots (GPTBot, ClaudeBot,
    <bot>_cum_share                  Google-Extended, Bytespider)
"""

import csv
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("E:/bot_paper_data")
INPUT_CSV = DATA_DIR / "daily_first_block_events_firmhostbot.csv"
GVKEY_OUT = DATA_DIR / "gvkey_first_block_dates.csv"
ADOPT_OUT = DATA_DIR / "gvkey_adoption_curve_monthly.csv"
SUMMARY_OUT = DATA_DIR / "gvkey_first_block_summary.txt"

GROUPS = ["search", "training", "assistant", "control"]
AI_GROUPS = ["search", "training", "assistant"]
HEADLINE_BOTS = ["GPTBot", "ClaudeBot", "Google-Extended", "Bytespider"]


def timing_quality(ival):
    if ival == "" or ival is None:
        return ""
    try:
        d = int(ival)
    except (ValueError, TypeError):
        return ""
    if d <= 7:
        return "le7"
    if d <= 14:
        return "le14"
    if d <= 30:
        return "le30"
    return "wide"


def yyyymm(ts14):
    if not ts14 or len(ts14) < 6:
        return ""
    return ts14[:6]


def add_month(ym, delta):
    """ym is 'YYYYMM' string; return ym + delta months as 'YYYYMM'."""
    y, m = int(ym[:4]), int(ym[4:6])
    total = y * 12 + (m - 1) + delta
    ny, nm = total // 12, total % 12 + 1
    return f"{ny:04d}{nm:02d}"


def main():
    print(f"Reading {INPUT_CSV}")

    # all_gvkeys: every gvkey in firmhostbot CSV (denominator for adoption)
    all_gvkeys = set()
    # per_pair[gvkey][bot] = (earliest_ts, interval_days, host)
    per_pair = defaultdict(dict)
    n_rows = 0
    n_with_event = 0

    with open(INPUT_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n_rows += 1
            gv = r["gvkey"]
            all_gvkeys.add(gv)
            fbt = r["first_observed_blocked_ts"]
            if not fbt:
                continue
            n_with_event += 1
            bot = r["bot"]
            host = r["host"]
            try:
                ival = int(r["interval_days"])
            except (ValueError, TypeError, KeyError):
                continue
            cur = per_pair[gv].get(bot)
            if cur is None or fbt < cur[0]:
                per_pair[gv][bot] = (fbt, ival, host)

    print(f"  rows scanned:                  {n_rows:,}")
    print(f"  rows with first_observed_blocked_ts: {n_with_event:,}")
    print(f"  unique gvkeys in firmhostbot:  {len(all_gvkeys):,}")
    print(f"  gvkeys with any event:         {len(per_pair):,}")

    # Per-bot taxonomy: build group lookup from one of the input rows
    print(f"\nLoading bot_group lookup...")
    bot_group_map = {}
    with open(INPUT_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            b = r["bot"]
            if b not in bot_group_map:
                bot_group_map[b] = r["bot_group"]
    print(f"  {len(bot_group_map)} bots, {len(set(bot_group_map.values()))} groups")

    # Bots-by-group for aggregation
    bots_by_group = defaultdict(list)
    for b, g in bot_group_map.items():
        bots_by_group[g].append(b)

    # ---- gvkey_first_block_dates.csv ----
    print(f"\nWriting {GVKEY_OUT}")
    fieldnames = ["gvkey"]
    for g in GROUPS:
        fieldnames += [f"first_{g}_block_ts", f"first_{g}_block_month",
                       f"interval_days_{g}", f"timing_quality_{g}",
                       f"first_{g}_block_bot", f"first_{g}_block_host"]
    fieldnames += ["first_any_ai_block_ts", "first_any_ai_block_month",
                   "first_any_ai_block_bot", "first_any_ai_block_host"]

    # Counters for summary
    n_event_per_group = {g: 0 for g in GROUPS}
    n_event_any_ai = 0

    with open(GVKEY_OUT, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for gv in sorted(all_gvkeys):
            pair = per_pair.get(gv, {})
            row = {"gvkey": gv}

            ai_min = None  # (ts, ival, bot, host)
            for g in GROUPS:
                grp_min = None
                for b in bots_by_group.get(g, []):
                    rec = pair.get(b)
                    if rec is None:
                        continue
                    if grp_min is None or rec[0] < grp_min[0]:
                        grp_min = (rec[0], rec[1], b, rec[2])
                if grp_min is None:
                    row[f"first_{g}_block_ts"] = ""
                    row[f"first_{g}_block_month"] = ""
                    row[f"interval_days_{g}"] = ""
                    row[f"timing_quality_{g}"] = ""
                    row[f"first_{g}_block_bot"] = ""
                    row[f"first_{g}_block_host"] = ""
                else:
                    ts, ival, bot, host = grp_min
                    row[f"first_{g}_block_ts"] = ts
                    row[f"first_{g}_block_month"] = yyyymm(ts)
                    row[f"interval_days_{g}"] = ival
                    row[f"timing_quality_{g}"] = timing_quality(ival)
                    row[f"first_{g}_block_bot"] = bot
                    row[f"first_{g}_block_host"] = host
                    n_event_per_group[g] += 1
                    if g in AI_GROUPS:
                        if ai_min is None or ts < ai_min[0]:
                            ai_min = (ts, ival, bot, host)

            if ai_min is None:
                row["first_any_ai_block_ts"] = ""
                row["first_any_ai_block_month"] = ""
                row["first_any_ai_block_bot"] = ""
                row["first_any_ai_block_host"] = ""
            else:
                ts, _, bot, host = ai_min
                row["first_any_ai_block_ts"] = ts
                row["first_any_ai_block_month"] = yyyymm(ts)
                row["first_any_ai_block_bot"] = bot
                row["first_any_ai_block_host"] = host
                n_event_any_ai += 1

            writer.writerow(row)

    print(f"  rows written: {len(all_gvkeys):,}")
    for g in GROUPS:
        print(f"    gvkeys with first_{g}_block_ts: {n_event_per_group[g]:,}")
    print(f"    gvkeys with first_any_ai_block_ts:   {n_event_any_ai:,}")

    # ---- gvkey_adoption_curve_monthly.csv ----
    print(f"\nWriting {ADOPT_OUT}")

    # New-events by month, per group and per headline bot
    new_by_month = {g: defaultdict(int) for g in GROUPS}
    new_by_month["any_ai"] = defaultdict(int)
    bot_new_by_month = {b: defaultdict(int) for b in HEADLINE_BOTS}

    # Iterate all gvkeys, count first-month adoptions per group / bot
    for gv in all_gvkeys:
        pair = per_pair.get(gv, {})
        # per group
        ai_min_ts = None
        for g in GROUPS:
            grp_min_ts = None
            for b in bots_by_group.get(g, []):
                rec = pair.get(b)
                if rec is None:
                    continue
                if grp_min_ts is None or rec[0] < grp_min_ts:
                    grp_min_ts = rec[0]
            if grp_min_ts is not None:
                new_by_month[g][yyyymm(grp_min_ts)] += 1
                if g in AI_GROUPS:
                    if ai_min_ts is None or grp_min_ts < ai_min_ts:
                        ai_min_ts = grp_min_ts
        if ai_min_ts is not None:
            new_by_month["any_ai"][yyyymm(ai_min_ts)] += 1

        # per headline bot
        for b in HEADLINE_BOTS:
            rec = pair.get(b)
            if rec is None:
                continue
            bot_new_by_month[b][yyyymm(rec[0])] += 1

    # Establish month range
    all_months = set()
    for d in new_by_month.values():
        all_months.update(d.keys())
    for d in bot_new_by_month.values():
        all_months.update(d.keys())
    all_months = [m for m in all_months if m]
    if not all_months:
        print("  no events; skipping adoption curve.")
        return
    first_m = min(all_months)
    last_m = max(all_months)
    months = []
    cur = first_m
    while cur <= last_m:
        months.append(cur)
        cur = add_month(cur, 1)

    n_at_risk = len(all_gvkeys)

    fields = ["month_yyyymm", "n_at_risk_gvkeys"]
    for g in GROUPS + ["any_ai"]:
        fields += [f"{g}_new_count", f"{g}_cum_count", f"{g}_cum_share"]
    for b in HEADLINE_BOTS:
        fields += [f"{b}_new_count", f"{b}_cum_count", f"{b}_cum_share"]

    cum = {g: 0 for g in (GROUPS + ["any_ai"])}
    bot_cum = {b: 0 for b in HEADLINE_BOTS}

    with open(ADOPT_OUT, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for m in months:
            row = {"month_yyyymm": m, "n_at_risk_gvkeys": n_at_risk}
            for g in (GROUPS + ["any_ai"]):
                nm = new_by_month[g].get(m, 0)
                cum[g] += nm
                row[f"{g}_new_count"] = nm
                row[f"{g}_cum_count"] = cum[g]
                row[f"{g}_cum_share"] = round(cum[g] / max(1, n_at_risk), 6)
            for b in HEADLINE_BOTS:
                nm = bot_new_by_month[b].get(m, 0)
                bot_cum[b] += nm
                row[f"{b}_new_count"] = nm
                row[f"{b}_cum_count"] = bot_cum[b]
                row[f"{b}_cum_share"] = round(bot_cum[b] / max(1, n_at_risk), 6)
            writer.writerow(row)
    print(f"  months: {len(months)}, range {months[0]}–{months[-1]}")
    print(f"  final cum shares:")
    for g in GROUPS + ["any_ai"]:
        print(f"    {g:<10} cum_count={cum[g]:>5,}  share={cum[g]/n_at_risk:.4f}")
    for b in HEADLINE_BOTS:
        print(f"    {b:<20} cum_count={bot_cum[b]:>5,}  share={bot_cum[b]/n_at_risk:.4f}")

    # ---- summary ----
    lines = []
    lines.append("gvkey_first_block aggregation summary")
    lines.append("")
    lines.append(f"Source:                                  {INPUT_CSV}")
    lines.append(f"  rows in firmhostbot:                   {n_rows:,}")
    lines.append(f"  rows with transition observed:         {n_with_event:,}")
    lines.append(f"  unique gvkeys in firmhostbot (at-risk):{n_at_risk:,}")
    lines.append("")
    lines.append(f"Output: {GVKEY_OUT}")
    lines.append(f"  rows: {len(all_gvkeys):,}  (one per gvkey)")
    for g in GROUPS:
        lines.append(f"  gvkeys with first_{g}_block_ts:       {n_event_per_group[g]:>6,}  "
                     f"({100*n_event_per_group[g]/n_at_risk:.2f}% of at-risk)")
    lines.append(f"  gvkeys with first_any_ai_block_ts:       {n_event_any_ai:>6,}  "
                 f"({100*n_event_any_ai/n_at_risk:.2f}% of at-risk)")
    lines.append("")
    lines.append(f"Output: {ADOPT_OUT}")
    lines.append(f"  rows: {len(months):,}  (one per calendar month, {months[0]}–{months[-1]})")
    lines.append(f"  final cumulative adoption shares (cum_count / at-risk):")
    for g in GROUPS + ["any_ai"]:
        lines.append(f"    {g:<10} {cum[g]:>5,}  ({100*cum[g]/n_at_risk:.2f}%)")
    for b in HEADLINE_BOTS:
        lines.append(f"    {b:<20} {bot_cum[b]:>5,}  ({100*bot_cum[b]/n_at_risk:.2f}%)")
    SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
