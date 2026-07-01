# plot_all.jl
# Julia equivalent of plot_all.py, using CairoMakie + AlgebraOfGraphics.
# Reads:  julia_descriptive/output/*.csv
# Writes: julia_descriptive/plots/figures/*.png
#
# One-time install:
#   using Pkg
#   Pkg.add(["CairoMakie", "AlgebraOfGraphics", "DataFrames", "CSV", "Dates", "Statistics"])
#
# Usage:
#   julia plot_all.jl                # generate all figures
#   julia plot_all.jl fig1 fig3      # generate specific figures

using CairoMakie
using AlgebraOfGraphics
using DataFrames, CSV, Dates, Statistics, Printf

# ============================================================
# Paths and style
# ============================================================
const HERE       = @__DIR__
const OUTPUT_DIR = joinpath(dirname(HERE), "output")
const FIG_DIR    = joinpath(HERE, "figures")
isdir(FIG_DIR) || mkpath(FIG_DIR)

# Publication-quality defaults
set_theme!(theme_minimal();
    fontsize          = 11,
    Axis              = (
        titlesize     = 12,
        xlabelsize    = 11,
        ylabelsize    = 11,
        xticklabelsize = 10,
        yticklabelsize = 10,
        topspinevisible    = false,
        rightspinevisible  = false,
    ),
    Legend = (
        framevisible  = false,
        labelsize     = 10,
    ),
    fonts             = (regular = "DejaVu Serif",),
)

# Colorblind-friendly palette (Wong)
const PALETTE = (
    blue   = RGBf(0/255, 114/255, 178/255),
    orange = RGBf(230/255, 159/255, 0/255),
    green  = RGBf(0/255, 158/255, 115/255),
    red    = RGBf(213/255, 94/255, 0/255),
    purple = RGBf(204/255, 121/255, 167/255),
    gray   = RGBf(120/255, 120/255, 120/255),
)
const COLOR_US      = PALETTE.red
const COLOR_NONUS   = PALETTE.blue
const COLOR_GPR     = PALETTE.green
const EXP_GROUP_COLORS = Dict(
    "ZERO" => PALETTE.gray,
    "LOW"  => PALETTE.blue,
    "MID"  => PALETTE.orange,
    "HIGH" => PALETTE.red,
)


# ============================================================
# Helpers
# ============================================================
function safe_read(filename::AbstractString)
    fp = joinpath(OUTPUT_DIR, filename)
    if !isfile(fp)
        @warn "missing: $filename"
        return nothing
    end
    df = CSV.read(fp, DataFrame)
    for c in (:month_end, :report_date, :month_first)
        if hasproperty(df, c)
            df[!, c] = Date.(df[!, c])
        end
    end
    return df
end

function date_axis!(ax, dates::AbstractVector{<:Date})
    isempty(dates) && return
    span_years = (maximum(dates) - minimum(dates)).value / 365.25
    # Adaptive step: aim for ~5-8 labels visible
    step = if span_years <= 6
        Year(1)
    elseif span_years <= 12
        Year(2)
    elseif span_years <= 30
        Year(5)
    else
        Year(10)
    end
    # Align ticks to multiples of step (e.g. 2000, 2005, 2010, ...)
    s = year(minimum(dates))
    e = year(maximum(dates))
    step_n = Dates.value(step)
    s_aligned = s - mod(s, step_n)  # round down to nearest multiple
    years = collect(Date(s_aligned, 1, 1):step:Date(e, 1, 1))
    # Keep only ticks within data range
    years = filter(y -> y >= Date(s, 1, 1) && y <= Date(e, 12, 31), years)
    isempty(years) && return
    ax.xticks = (Dates.value.(years), string.(year.(years)))
end

function _save(fig, name::AbstractString)
    out = joinpath(FIG_DIR, "$name.png")
    save(out, fig; px_per_unit=2)  # 2x for higher resolution
    rel = relpath(out, dirname(HERE))
    println("  -> $rel")
end

# Convert Date -> numeric for Makie scatter/lines axes (Makie handles Date but
# manual conversion is reliable across versions)
to_num(d::Date) = Dates.value(d)
to_num(d::AbstractVector{<:Date}) = to_num.(d)

# Thousand-separator formatter for axis labels (avoids label truncation when
# numbers are large, e.g. 1,654,063 instead of bare "1654063" that may get
# clipped at axis edge)
fmt_thousand(v::Integer) = replace(string(v), r"(?<=\d)(?=(\d{3})+(?!\d))" => ",")
fmt_thousand(v::Real)    = fmt_thousand(round(Int, v))


# ============================================================
# FIG 1: China-exposure overview (2x2)
# ============================================================
function fig1_china_exposure_overview()
    ts   = safe_read("02_china_exposure_timeseries.csv")
    dist = safe_read("02_dist_cn_total_2018.csv")
    ts === nothing && return

    fig = Figure(size=(1200, 800))

    # (a) # EU firms with any CN supply-chain link
    ax1 = Axis(fig[1,1],
        title="(a) # EU firms with CN supply-chain link",
        xlabel="", ylabel="# firms")
    lines!(ax1, to_num(ts.month_end), ts.n_firms_with_cn,
           color=PALETTE.blue, linewidth=1.8)
    date_axis!(ax1, ts.month_end)

    # (b) Mean China share of supply-chain links (advisor-revised exposure)
    ax2 = Axis(fig[1,2],
        title="(b) Mean China share of supply-chain links",
        xlabel="", ylabel="share (China-links / total links)")
    if hasproperty(ts, :avg_china_share_among_exposed)
        lines!(ax2, to_num(ts.month_end), ts.avg_china_share_among_exposed;
               color=PALETTE.red, linewidth=1.8, label="among CN-exposed firms")
        if hasproperty(ts, :avg_china_share)
            lines!(ax2, to_num(ts.month_end), ts.avg_china_share;
                   color=PALETTE.orange, linewidth=1.5, linestyle=:dash,
                   label="all firms with any link")
        end
        axislegend(ax2; position=:lt, framevisible=false)
    elseif hasproperty(ts, :avg_cn_rels)
        lines!(ax2, to_num(ts.month_end), ts.avg_cn_rels;
               color=PALETTE.red, linewidth=1.8)
        ax2.title = "(b) Avg # CN relations per exposed firm"
        ax2.ylabel = "mean count"
    end
    date_axis!(ax2, ts.month_end)

    # (c) Distribution at snapshot
    ax3 = Axis(fig[2,1],
        title="(c) Distribution of # CN relations per exposed firm (2018-12-31)",
        xlabel="# CN relations", ylabel="# firms")
    if dist !== nothing && nrow(dist) > 0
        max_n = min(40, maximum(dist.n_cn_total) + 2)
        barplot!(ax3, dist.n_cn_total, dist.n_firms,
                 color=(PALETTE.blue, 0.75), strokecolor=:white, strokewidth=0.5)
        xlims!(ax3, 0, max_n)
    else
        text!(ax3, 0.5, 0.5; text="snapshot CSV not found", align=(:center, :center))
    end

    # (d) Breakdown by rel_type — raw Makie (so legend works cleanly)
    ax4 = Axis(fig[2,2],
        title="(d) Relation type composition over time",
        xlabel="", ylabel="avg # per exposed firm")
    type_specs = [
        (:avg_cn_customer, "CUSTOMER", PALETTE.blue),
        (:avg_cn_supplier, "SUPPLIER", PALETTE.orange),
        (:avg_cn_jv,       "JV",       PALETTE.red),
    ]
    for (col, label, color) in type_specs
        if hasproperty(ts, col)
            lines!(ax4, to_num(ts.month_end), ts[!, col];
                   color=color, linewidth=1.5, label=label)
        end
    end
    if hasproperty(ts, :avg_cn_rels)
        lines!(ax4, to_num(ts.month_end), ts.avg_cn_rels;
               color=:black, linewidth=2, linestyle=:dash, label="TOTAL")
    end
    axislegend(ax4; position=:lt, nbanks=2, framevisible=false)
    date_axis!(ax4, ts.month_end)

    Label(fig[0, :], "China supply-chain exposure of European firms (Revere)",
          fontsize=14, font=:bold)

    _save(fig, "fig1_cn_exposure_overview")
end


# ============================================================
# FIG 2: US-EU engagement (2 panels vertical, dual axis top)
# ============================================================
function fig2_us_eu_engagement()
    df = safe_read("03_us_x_eu_cells_by_month.csv")
    df === nothing && return

    fig = Figure(size=(1100, 700))

    # Panel A: counts with dual y-axis
    ax1 = Axis(fig[1,1],
        title="(a) US-EU engagement: investor and firm counts",
        xlabel="", ylabel="# US funds",
        ylabelcolor=COLOR_US, yticklabelcolor=COLOR_US)
    l1 = lines!(ax1, to_num(df.report_date), df.n_us_funds,
                color=COLOR_US, linewidth=1.8)
    date_axis!(ax1, df.report_date)

    # Twin y-axis: # EU firms
    ax1b = Axis(fig[1,1]; ylabel="# EU firms", yaxisposition=:right,
                ylabelcolor=COLOR_NONUS, yticklabelcolor=COLOR_NONUS,
                rightspinevisible=true)
    hidespines!(ax1b, :t, :b, :l)
    hidexdecorations!(ax1b)
    l2 = lines!(ax1b, to_num(df.report_date), df.n_eu_firms,
                color=COLOR_NONUS, linewidth=1.8, linestyle=:dash)

    # Legend OUTSIDE the axis area to avoid overlapping the highly volatile
    # quarter-end vs off-quarter reporting pattern.
    axislegend(ax1, [l1, l2], ["# US funds", "# EU firms"];
               position=:lt, orientation=:horizontal, framevisible=true,
               backgroundcolor=(:white, 0.85))

    # Panel B: total MV. Column name varies between 03 versions —
    # total_mv_billions (current 03) vs total_mv_b (earlier draft).
    ax2 = Axis(fig[2,1],
        title="(b) Total US-held EU equity (USD billions)",
        xlabel="", ylabel="USD billions")
    mv_col = hasproperty(df, :total_mv_billions) ? :total_mv_billions : :total_mv_b
    lines!(ax2, to_num(df.report_date), df[!, mv_col],
           color=PALETTE.green, linewidth=1.8)
    date_axis!(ax2, df.report_date)

    Label(fig[0, :], "US institutional engagement in European equities",
          fontsize=14, font=:bold)
    _save(fig, "fig2_us_eu_engagement")
end


# ============================================================
# FIG 3: US-ownership distribution + by-country (2x2)
# ============================================================
function fig3_us_ownership_distribution()
    snap = safe_read("04_us_ownership_eu_snapshot.csv")
    bycn = safe_read("04_us_own_by_eu_country_snapshot.csv")
    ts   = safe_read("04_us_ownership_eu_timeseries.csv")
    if snap === nothing && bycn === nothing && ts === nothing
        return
    end

    fig = Figure(size=(1300, 850))

    # (a) Distribution
    # Two issues with the naive histogram:
    #   (i) ~850 firms cluster near 0% (token positions), dwarfing the tail.
    #   (ii) Percentile lines on the 0-spike are visually buried.
    # Fix: filter to firms with ownership > 0.5% (i.e. drop the token cluster
    # below half a percent), so the meaningful distribution becomes visible.
    # Report the dropped-N alongside so the reader sees the trim.
    ax1 = Axis(fig[1,1],
        title="(a) Distribution of US ownership share, EU firms (snapshot)",
        xlabel="US ownership share (%)", ylabel="# firms")
    if snap !== nothing && nrow(snap) > 0
        x_all = collect(skipmissing(snap.ownership_share)) .* 100
        n_dropped = count(<(0.5), x_all)
        x = filter(>=(0.5), x_all)
        hist!(ax1, x, bins=40, color=(PALETTE.blue, 0.75), strokewidth=0.5,
              strokecolor=:white)
        xlim_top = min(40, quantile(x, 0.99) * 1.1)
        xlims!(ax1, 0, xlim_top)
        # Percentiles computed on the FULL distribution (including the
        # filtered tail) so they describe the actual sample, not the trim.
        p50 = quantile(x_all, 0.50)
        p75 = quantile(x_all, 0.75)
        p90 = quantile(x_all, 0.90)
        vlines!(ax1, [p50]; color=:black,         linestyle=:dash, linewidth=1)
        vlines!(ax1, [p75]; color=PALETTE.gray,   linestyle=:dash, linewidth=1)
        vlines!(ax1, [p90]; color=PALETTE.red,    linestyle=:dash, linewidth=1)
        text!(ax1, 0.98, 0.96;
              text = "P50 = $(round(p50, digits=1))%\n" *
                     "P75 = $(round(p75, digits=1))%\n" *
                     "P90 = $(round(p90, digits=1))%\n" *
                     "(N=$(length(x_all)); $n_dropped firms <0.5% trimmed from view)",
              align=(:right, :top), space=:relative, fontsize=9)
    end

    # (b) Mean over time
    ax2 = Axis(fig[1,2],
        title="(b) Mean US ownership of EU firms over time",
        xlabel="", ylabel="mean ownership (%)")
    if ts !== nothing && nrow(ts) > 0
        lines!(ax2, to_num(ts.report_date), ts.mean_us_own .* 100,
               color=COLOR_US, linewidth=1.8)
        date_axis!(ax2, ts.report_date)
    end

    # (c) Total USD over time
    ax3 = Axis(fig[2,1],
        title="(c) Total US-held EU equity over time (USD B)",
        xlabel="", ylabel="USD billions")
    if ts !== nothing && hasproperty(ts, :total_us_holding_b)
        lines!(ax3, to_num(ts.report_date), ts.total_us_holding_b,
               color=PALETTE.green, linewidth=1.8)
        date_axis!(ax3, ts.report_date)
    end

    # (d) By country, horizontal bar
    # Makie barplot direction=:x: first arg = positions (Y axis), second = values (X)
    ax4 = Axis(fig[2,2],
        title="(d) Total US-held EU equity by country (USD B, top 15)",
        xlabel="USD billions", ylabel="")
    if bycn !== nothing && nrow(bycn) > 0
        sorted = sort(bycn, :total_us_holding_b)
        top = last(sorted, 15)
        n = nrow(top)
        ypos = collect(1:n)
        barplot!(ax4, ypos, top.total_us_holding_b;
                 color=(PALETTE.blue, 0.85), strokecolor=:white, strokewidth=0.5,
                 direction=:x)
        ax4.yticks = (ypos, top.sec_country)
        for (i, v) in enumerate(top.total_us_holding_b)
            text!(ax4, v, i; text=" " * fmt_thousand(round(Int, v)),
                  align=(:left, :center), fontsize=9)
        end
        xlims!(ax4, 0, maximum(top.total_us_holding_b) * 1.20)
    end

    Label(fig[0, :], "US institutional ownership of European firms",
          fontsize=14, font=:bold)
    _save(fig, "fig3_us_ownership_distribution")
end


# ============================================================
# FIG 4: Pre-regression (2x2)
# ============================================================
function fig4_pre_regression()
    sc   = safe_read("05_scatter_own_vs_cn_data.csv")
    ts   = safe_read("05_within_europe_share_by_group.csv")
    # cmp = 05_us_vs_nonus_high_share_data.csv was used by the old B' path
    # which mixed US-within-EU-share with NONUS-global-share. The new B' reads
    # abs_portfolio_weight directly from ts (both US and NONUS rows present),
    # so cmp is no longer needed here.
    diff = safe_read("05_diff_us_vs_nonus_high.csv")

    fig = Figure(size=(1300, 900))

    # (A) Scatter ownership vs CN
    # Advisor revision: use china_share (share of supply-chain links) if
    # available; fall back to log(1 + count) otherwise.
    use_share_A = sc !== nothing && hasproperty(sc, :china_share) &&
                  count(!ismissing, sc.china_share) > 5
    xlabel_A = use_share_A ? "China share of supply-chain links" :
                              "log(1 + # CN supply-chain relations)"
    ax1 = Axis(fig[1,1],
        title="(A) US ownership vs China exposure",
        xlabel=xlabel_A,
        ylabel="US ownership (%)")
    if sc !== nothing && nrow(sc) > 5
        x_raw = use_share_A ? sc.china_share : (log.(1 .+ sc.n_cn_total))
        y_raw = sc.us_ownership_share
        keep = .!ismissing.(x_raw) .& .!ismissing.(y_raw)
        x = Float64.(x_raw[keep])
        y = Float64.(y_raw[keep]) .* 100
        scatter!(ax1, x, y, color=(PALETTE.blue, 0.35), markersize=4)
        if length(x) > 1 && std(x) > 0
            m = cor(x, y) * std(y) / std(x)
            b = mean(y) - m * mean(x)
            if !ismissing(m) && !ismissing(b)
                xs = range(minimum(x), maximum(x), length=100)
                lines!(ax1, xs, m .* xs .+ b, color=:black, linewidth=1.2,
                       linestyle=:dash, label="slope=$(round(m, digits=2))")
                axislegend(ax1; position=:rt)
            end
        end
    end

    # (B) Within-Europe share by exposure group (US only)
    # Raw Makie (not AoG) so axislegend works cleanly.
    # CAVEAT: Revere supply-chain data starts 2003-Q1. Pre-2003 quarters have
    # no exposure information, so every firm gets classified as ZERO by
    # default — this artifactually inflates the ZERO line for 1999-2002.
    # We clip the time series to 2003-Q1 onward and mark the start visually.
    REVERE_START = Date(2003, 3, 31)
    if ts !== nothing && nrow(ts) > 0
        ax_B = Axis(fig[1,2],
            title="(B) US allocation by exposure group",
            xlabel="", ylabel="US portfolio weight on EU group (basis points)")
        us = filter(r -> r.investor_country == "US" && r.report_date >= REVERE_START, ts)
        # Plot abs_portfolio_weight (the true Section 3 measure), in basis
        # points (× 10,000) so the axis is readable. This is "fraction of US
        # global institutional portfolio allocated to this exposure group".
        for grp in ["ZERO", "LOW", "MID", "HIGH"]
            sub = filter(:exp_grp => ==(grp), us)
            keep = .!ismissing.(sub.abs_portfolio_weight)
            sum(keep) == 0 && continue
            lines!(ax_B, to_num(sub.report_date[keep]),
                   identity.(sub.abs_portfolio_weight[keep]) .* 10_000;
                   color=EXP_GROUP_COLORS[grp], linewidth=1.8, label=grp)
        end
        axislegend(ax_B; position=:rt, framevisible=false,
                   labelsize=10, title="exposure", titlesize=10)
        date_axis!(ax_B, us.report_date)
    end

    # (B') US vs non-US ABSOLUTE portfolio weight on HIGH-exposure firms.
    # Both lines use abs_portfolio_weight (fraction of investor's GLOBAL
    # portfolio allocated to HIGH-CN-exposed EU firms), in basis points,
    # so US and NONUS are on the same scale (single denominator each).
    ax3 = Axis(fig[2,1],
        title="(B') Portfolio weight on HIGH-exposure firms",
        xlabel="", ylabel="portfolio weight (basis points)")
    if ts !== nothing
        # Clip to 2003-Q1 onward (Revere coverage start) to avoid mis-bucketing
        # pre-2003 quarters as ZERO/LOW/MID/HIGH when no exposure data exists.
        us_high    = filter(r -> r.investor_country == "US"    && r.exp_grp == "HIGH" &&
                                 r.report_date >= REVERE_START, ts)
        nonus_high = filter(r -> r.investor_country == "NONUS" && r.exp_grp == "HIGH" &&
                                 r.report_date >= REVERE_START, ts)
        if nrow(us_high) > 0
            keep_us = .!ismissing.(us_high.abs_portfolio_weight)
            if sum(keep_us) > 0
                lines!(ax3, to_num(us_high.report_date[keep_us]),
                       Float64.(us_high.abs_portfolio_weight[keep_us]) .* 10_000;
                       color=COLOR_US, linewidth=1.8, label="US")
                date_axis!(ax3, us_high.report_date[keep_us])
            end
        end
        if nrow(nonus_high) > 0
            keep_n = .!ismissing.(nonus_high.abs_portfolio_weight)
            if sum(keep_n) > 0
                lines!(ax3, to_num(nonus_high.report_date[keep_n]),
                       Float64.(nonus_high.abs_portfolio_weight[keep_n]) .* 10_000;
                       color=COLOR_NONUS, linewidth=1.8, label="non-US")
            end
        end
        axislegend(ax3; position=:rt)

        # GPR on right axis — use ts (has NONUS rows with gpr_us_cn) so we
        # don't depend on the legacy cmp DataFrame. Also clip to Revere start.
        gpr_rows = filter(r -> r.investor_country == "NONUS" && r.exp_grp == "HIGH" &&
                               !ismissing(r.gpr_us_cn) &&
                               r.report_date >= REVERE_START, ts)
        if nrow(gpr_rows) > 0
            ax3b = Axis(fig[2,1]; ylabel="USA|China GPR",
                        yaxisposition=:right,
                        ylabelcolor=COLOR_GPR, yticklabelcolor=COLOR_GPR,
                        rightspinevisible=true)
            hidespines!(ax3b, :t, :b, :l)
            hidexdecorations!(ax3b)
            lines!(ax3b, to_num(gpr_rows.report_date),
                   Float64.(gpr_rows.gpr_us_cn);
                   color=(COLOR_GPR, 0.6), linewidth=1, linestyle=:dot)
        end
    end

    # (C) Differential vs GPR scatter
    # Advisor revision: x-axis is the AR(1) shock of bilateral GPR, not the
    # level. Falls back to level if shock not available.
    use_shock = diff !== nothing && hasproperty(diff, :shock_us_cn) &&
                count(!ismissing, diff.shock_us_cn) > 5
    ax4_xlabel = use_shock ? "USA|China GPR AR(1) shock" : "USA|China GPR level"
    ax4_title  = use_shock ? "(C) ΔUS − ΔnonUS vs GPR shock, HIGH-exposure firms" :
                              "(C) ΔUS − ΔnonUS vs GPR level, HIGH-exposure firms"
    ax4 = Axis(fig[2,2], title=ax4_title, xlabel=ax4_xlabel,
        ylabel="quarterly Δ portfolio weight, US − non-US (basis points)")
    if diff !== nothing && nrow(diff) > 5
        # mean_diff is Δw (absolute portfolio-weight change), scaled to basis
        # points (× 10,000) for readable axis. Drop any rows where the chosen
        # X or Y is missing to avoid Vector{Missing} propagation through fit.
        x_raw = use_shock ? diff.shock_us_cn : diff.gpr_us_cn
        y_raw = diff.mean_diff
        keep = .!ismissing.(x_raw) .& .!ismissing.(y_raw)
        # Force concrete Float64 type so std/cor don't propagate Missing typing
        x = Float64.(x_raw[keep])
        y = Float64.(y_raw[keep]) .* 10_000
        scatter!(ax4, x, y, color=(PALETTE.red, 0.6), markersize=8)
        hlines!(ax4, [0], color=PALETTE.gray, linewidth=0.7)
        if length(x) > 1 && std(x) > 0
            m = cor(x, y) * std(y) / std(x)
            b = mean(y) - m * mean(x)
            xs = range(minimum(x), maximum(x), length=100)
            if !ismissing(m) && !ismissing(b)
                lines!(ax4, xs, m .* xs .+ b; color=:black, linewidth=1.2,
                       linestyle=:dash, label="slope=$(round(m, digits=3))")
            end
            corr_val = cor(x, y)
            text!(ax4, 0.04, 0.96;
                  text="corr = $(round(corr_val, digits=3, base=10))",
                  space=:relative, align=(:left, :top), fontsize=10)
            axislegend(ax4; position=:rt)
        end
    end

    Label(fig[0, :], "US holdings, China exposure, and bilateral GPR",
          fontsize=14, font=:bold)
    Label(fig[-1, :],
          "Panels B and B′ clipped to 2003-Q1 onward (start of Revere supply-chain coverage)";
          fontsize=10, color=:gray)
    _save(fig, "fig4_pre_regression")
end


# ============================================================
# FIG 5: Coverage diagnostics
# ============================================================
function fig5_coverage()
    cas   = safe_read("05_coverage_cascade.csv")
    comp  = safe_read("05_merged_panel_composition.csv")
    mtype = safe_read("05_match_type_distribution.csv")

    fig = Figure(size=(1500, 500))

    # (A) Cascade
    ax1 = Axis(fig[1,1],
        title="(A) Coverage cascade: EU sec_entity → matched Revere",
        ylabel="# sec_entity_ids")
    if cas !== nothing && nrow(cas) > 0
        row = cas[1, :]
        labels = String[]
        vals = Int[]
        for col in names(cas)
            push!(labels, replace(replace(col, "n_eu_sec_" => ""), "_" => "\n"))
            push!(vals, row[col])
        end
        ax1.xticks = (1:length(labels), labels)
        ax1.xticklabelrotation = 0
        # Color palette extends to as many columns as needed (cyclic if longer)
        base = [PALETTE.blue, PALETTE.orange, PALETTE.green, PALETTE.red, PALETTE.purple, PALETTE.gray]
        colors = [base[((i-1) % length(base)) + 1] for i in 1:length(labels)]
        barplot!(ax1, 1:length(labels), vals;
                 color=colors, strokecolor=:white, strokewidth=0.5)
        for (i, v) in enumerate(vals)
            text!(ax1, i, v; text=fmt_thousand(v),
                  align=(:center, :bottom), fontsize=10)
        end
        # Add 10% headroom so labels don't get clipped at top
        ylims!(ax1, 0, maximum(vals) * 1.10)
    end

    # (B) Merged panel composition
    ax2 = Axis(fig[1,2],
        title="(B) Matched panel composition",
        xlabel="# (h, i, t) observations")
    if comp !== nothing && nrow(comp) > 0
        comp_sorted = sort(comp, :n)
        labels = String[]
        colors = RGBf[]
        for r in eachrow(comp_sorted)
            us_label = r.is_us ? "US " : "non-US "
            cn_label = r.has_cn_exposure ? "with CN" : "no CN"
            push!(labels, us_label * cn_label)
            base = r.is_us ? COLOR_US : COLOR_NONUS
            push!(colors, r.has_cn_exposure ? base : RGBf(0.8, 0.8, 0.8))
        end
        n = length(labels)
        barplot!(ax2, 1:n, comp_sorted.n;
                 color=colors, strokecolor=:white, strokewidth=0.5,
                 direction=:x)
        ax2.yticks = (1:n, labels)
        for (i, v) in enumerate(comp_sorted.n)
            text!(ax2, v, i; text=" " * fmt_thousand(v),
                  align=(:left, :center), fontsize=9)
        end
        # Right-margin so big labels (e.g. 1,654,063) don't get clipped
        xlims!(ax2, 0, maximum(comp_sorted.n) * 1.20)
    end

    # (C) Match type
    ax3 = Axis(fig[1,3],
        title="(C) ID type contribution to matching",
        xlabel="# sec_entity_ids matched")
    if mtype !== nothing && nrow(mtype) > 0
        mtype_sorted = sort(mtype, :n_sec_entities)
        n = nrow(mtype_sorted)
        barplot!(ax3, 1:n, mtype_sorted.n_sec_entities;
                 color=(PALETTE.blue, 0.85), strokecolor=:white, strokewidth=0.5,
                 direction=:x)
        ax3.yticks = (1:n, mtype_sorted.match_type)
        for (i, v) in enumerate(mtype_sorted.n_sec_entities)
            text!(ax3, v, i; text=" " * fmt_thousand(v),
                  align=(:left, :center), fontsize=9)
        end
        xlims!(ax3, 0, maximum(mtype_sorted.n_sec_entities) * 1.20)
    end

    Label(fig[0, :], "Sample-coverage diagnostics",
          fontsize=14, font=:bold)
    _save(fig, "fig5_coverage_diagnostics")
end


# ============================================================
# Dispatch
# ============================================================
const FIGURES = Dict(
    "fig1" => fig1_china_exposure_overview,
    "fig2" => fig2_us_eu_engagement,
    "fig3" => fig3_us_ownership_distribution,
    "fig4" => fig4_pre_regression,
    "fig5" => fig5_coverage,
)

function main()
    names = isempty(ARGS) ? collect(keys(FIGURES)) : ARGS
    for name in names
        if !haskey(FIGURES, name)
            println("unknown figure: $name; valid: $(collect(keys(FIGURES)))")
            continue
        end
        println("\n[$name]")
        try
            FIGURES[name]()
        catch e
            println("  ERROR: $e")
            for (i, frame) in enumerate(stacktrace(catch_backtrace()))
                i > 8 && break
                println("    $frame")
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
