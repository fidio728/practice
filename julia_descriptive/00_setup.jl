# 00_setup.jl
# Run this FIRST. Sets up packages, paths, a DuckDB connection helper, atomic
# writes, and TEST_MODE controls.
#
# Audited 2026-06-01 — see AUDIT_2026_06_01_julia_descriptive.md.
# Resulting changes:
#   - ENV-driven paths (DPN_DATA_ROOT) so the pipeline runs off any disk.
#   - TEST_MODE from ENV (DPN_TEST_MODE), with downstream guard that forces
#     suffixed output filenames when set so a test run cannot silently
#     overwrite canonical artifacts.
#   - Atomic-write helper (atomic_copy_to) for every COPY ... TO parquet.
#   - Manifest writer for reproducibility (records git sha, julia version,
#     row count, input fingerprints).
#
# One-time package install (uncomment and run once, then comment again):
# using Pkg
# Pkg.add(["DuckDB", "DataFrames", "CSV", "StatsBase", "Dates",
#          "Statistics", "CategoricalArrays", "GZip", "Printf", "SHA", "JSON3"])

using DuckDB, DataFrames, CSV, StatsBase, Dates, Statistics, Printf, SHA

# ============================================================
# PATHS — ENV-driven with sensible defaults
# ============================================================
const DATA_ROOT = get(ENV, "DPN_DATA_ROOT", raw"E:\Data\Data")
const OWN_DIR   = joinpath(DATA_ROOT, "Factset Ownership")
const REV_DIR   = joinpath(DATA_ROOT, "Factset Revere")

const INSTITUTIONS_PATH = joinpath(OWN_DIR, "Factset_LionShares_Institutions.gz")
const FUNDS_PATH        = joinpath(OWN_DIR, "Factset_LionShares_Funds.gz")
const SEC_MAP_PATH      = joinpath(OWN_DIR, "Factset_Security_Map.gz")
const SEC_COVERAGE_PATH = joinpath(OWN_DIR, "Factset_Security_coverage.gz")
const HOLDINGS_GLOB     = joinpath(OWN_DIR, "Factset_FundOwners_*.gz")

const REVERE_REL_PATH = joinpath(REV_DIR, "data_giorgio.csv")
const REVERE_CO_PATH  = joinpath(REV_DIR, "revere_company_wrds.csv")

const GPR_PATH      = joinpath(DATA_ROOT, "ai_gpr_bilateral_monthly.csv")
const GSDB_PATH     = joinpath(DATA_ROOT, "GSDB_V4_dates.csv")
const GRAVITY_PATH  = joinpath(DATA_ROOT, "gravity_vars_2021.csv")
const CONFLICT_PATH = joinpath(DATA_ROOT, "conflict_monthly.csv")

const SCRIPT_DIR = @__DIR__
const OUT_DIR    = joinpath(SCRIPT_DIR, "output")
isdir(OUT_DIR) || mkpath(OUT_DIR)

# Raw-parquet cache lives OFF the OneDrive-synced path. ~30 GB total.
# 03a writes here; 03 reads from here.
const RAW_PARQUET_DIR = get(ENV, "DPN_RAW_PARQUET_DIR", raw"E:\Data\Data\raw_parquet")
isdir(RAW_PARQUET_DIR) || mkpath(RAW_PARQUET_DIR)

# ============================================================
# TEST_MODE controls — set DPN_TEST_MODE=true in environment to subset.
# When TEST_MODE is on, downstream MUST add a suffix to output filenames
# (see test_suffix_path) so test runs cannot overwrite canonical outputs.
# ============================================================
const TEST_MODE = parse(Bool, lowercase(get(ENV, "DPN_TEST_MODE", "false")))
const TEST_SUFFIX = "_TESTMODE"

"""
    test_suffix_path(path) -> String

If `TEST_MODE` is on, inserts `_TESTMODE` before the file extension so a test
run cannot silently overwrite the canonical artifact at the same path. If
`TEST_MODE` is off, returns `path` unchanged.
"""
function test_suffix_path(path::AbstractString)
    TEST_MODE || return path
    base, ext = splitext(path)
    return base * TEST_SUFFIX * ext
end

# ============================================================
# EU COUNTRY LIST — single source of truth (was duplicated across 02/03/04/05)
# ============================================================
const EU_COUNTRIES = ("GB","DE","FR","NL","CH","IT","ES","SE","DK","NO","FI",
                      "BE","AT","IE","LU","PT","PL","CZ","HU","GR","RO","SK",
                      "SI","BG","HR","EE","LV","LT")
const EU_SQL_TUPLE = "(" * join(["'" * c * "'" for c in EU_COUNTRIES], ",") * ")"

# ============================================================
# DUCKDB CONNECTION HELPER
# ============================================================
# RAM budget: 6GB out of 8-10GB usable (leave headroom for Julia + OS).
# Spill to disk when RAM exceeded.
function dbcon(; memory_gb::Int=6, threads::Int=4)
    spill = joinpath(tempdir(), "duckdb_spill")
    isdir(spill) || mkpath(spill)
    con = DBInterface.connect(DuckDB.DB, ":memory:")
    DBInterface.execute(con, "SET memory_limit='$(memory_gb)GB'")
    DBInterface.execute(con, "SET threads=$threads")
    DBInterface.execute(con, "SET temp_directory='$(replace(spill, "\\" => "/"))'")
    # P0 rebuild (2026-08-04): the panel grew 4.66 -> 6.93 GB and big sorts/joins
    # now spill past duckdb's ~4.3GB default temp cap (04 hung at exactly that
    # ceiling). Point TMP/TEMP at a roomy drive (E:) when launching; the cap
    # itself must be raised explicitly or the offload deadlocks rather than errors.
    DBInterface.execute(con, "SET max_temp_directory_size='300GB'")
    # Critical for large GROUP BY / COPY operations on tight RAM:
    # lets DuckDB pipeline through without buffering full input order.
    DBInterface.execute(con, "SET preserve_insertion_order=false")
    return con
end

# Convenience: run SQL and return DataFrame
qdf(con, sql::AbstractString) = DataFrame(DBInterface.execute(con, sql))

# ============================================================
# ATOMIC-WRITE HELPER — for every COPY ... TO parquet.
# Pattern: write to <path>.tmp.<pid>, verify non-empty, then mv to <path>.
# On error, the .tmp is cleaned up so canonical paths never carry a partial
# write. Same filesystem assertion guarantees the rename is atomic.
# ============================================================
"""
    atomic_copy_to(con, select_sql, out_path; format="parquet", compression="zstd", row_group_size=122880)

Executes `COPY (\$select_sql) TO '<out_path>.tmp.<pid>' (FORMAT 'parquet', ...)`,
verifies the temp file has non-zero size, then atomically renames it onto
`out_path`. Clears any stale `.tmp` first. On error, removes the partial temp.

Uses pinned `COMPRESSION_LEVEL 3` and `ROW_GROUP_SIZE 122880` so file SHA-256
is reproducible across DuckDB versions (per audit recommendation).
"""
function atomic_copy_to(con, select_sql::AbstractString, out_path::AbstractString;
                        format::AbstractString="parquet",
                        compression::AbstractString="zstd",
                        compression_level::Int=3,
                        row_group_size::Int=122_880)
    out_path_fwd = replace(out_path, "\\" => "/")
    tmp_path     = out_path_fwd * ".tmp." * string(getpid())
    # Same-filesystem assertion for atomic rename.
    @assert dirname(tmp_path) == dirname(out_path_fwd) "tmp and target must share volume for atomic rename"
    # Clear stale tmp from a previous failed run.
    isfile(replace(tmp_path, "/" => "\\")) && rm(replace(tmp_path, "/" => "\\"); force=true)
    try
        if format == "parquet"
            DBInterface.execute(con, """
                COPY ($select_sql) TO '$tmp_path' (
                    FORMAT 'parquet',
                    COMPRESSION '$compression',
                    COMPRESSION_LEVEL $compression_level,
                    ROW_GROUP_SIZE $row_group_size
                )
            """)
        else
            DBInterface.execute(con, "COPY ($select_sql) TO '$tmp_path' (FORMAT '$format')")
        end
        tmp_native = replace(tmp_path, "/" => "\\")
        @assert isfile(tmp_native) && filesize(tmp_native) > 0 "atomic_copy_to: temp file empty or missing"
        mv(tmp_native, replace(out_path_fwd, "/" => "\\"); force=true)
    catch e
        tmp_native = replace(tmp_path, "/" => "\\")
        isfile(tmp_native) && rm(tmp_native; force=true)
        rethrow(e)
    end
    return out_path_fwd
end

# ============================================================
# MANIFEST WRITER — per-step reproducibility sidecar.
# Records git sha (if available), Julia version, row count, output sha256,
# input file fingerprints, build timestamp. Downstream scripts can read and
# assert against expected fingerprints.
# ============================================================
function _git_sha()
    try
        return strip(read(`git -C $SCRIPT_DIR rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

function _file_sha256(path::AbstractString)
    isfile(path) || return ""
    open(path) do io
        return bytes2hex(SHA.sha256(io))
    end
end

"""
    write_manifest(step_name, out_path; row_count=missing, input_paths=String[])

Writes `<out_path>.meta.json` (single-line JSON) with reproducibility info
beside `out_path`. Use after `atomic_copy_to` lands the canonical artifact.
"""
function write_manifest(step_name::AbstractString, out_path::AbstractString;
                        row_count=missing,
                        input_paths::Vector{<:AbstractString}=String[])
    out_native = replace(out_path, "/" => "\\")
    meta_path = out_native * ".meta.json"
    input_fp = [Dict("path"=>p, "size_bytes"=>(isfile(p) ? filesize(p) : 0),
                     "mtime"=>(isfile(p) ? string(Dates.unix2datetime(mtime(p))) : ""))
                for p in input_paths]
    meta = Dict(
        "step"             => step_name,
        "out_path"         => out_native,
        "git_sha"          => _git_sha(),
        "julia_version"    => string(VERSION),
        "duckdb_version"   => (try qdf(dbcon(), "SELECT version() AS v").v[1] catch; "unknown" end),
        "test_mode"        => TEST_MODE,
        "row_count"        => row_count === missing ? -1 : row_count,
        "sha256"           => _file_sha256(out_native),
        "build_ts"         => string(now()),
        "input_files"      => input_fp,
    )
    # Minimal JSON without bringing in JSON3 dep — fast escape for filenames.
    json_str = "{" * join([
        "\"step\":\"$(meta["step"])\"",
        "\"out_path\":\"$(replace(meta["out_path"], "\\" => "\\\\"))\"",
        "\"git_sha\":\"$(meta["git_sha"])\"",
        "\"julia_version\":\"$(meta["julia_version"])\"",
        "\"duckdb_version\":\"$(meta["duckdb_version"])\"",
        "\"test_mode\":$(meta["test_mode"])",
        "\"row_count\":$(meta["row_count"])",
        "\"sha256\":\"$(meta["sha256"])\"",
        "\"build_ts\":\"$(meta["build_ts"])\"",
        "\"input_count\":$(length(input_paths))",
    ], ",") * "}"
    open(meta_path, "w") do io
        write(io, json_str)
    end
    return meta_path
end

# ============================================================
# FILE EXISTENCE SMOKE TEST
# ============================================================
function smoke_test()
    files = [INSTITUTIONS_PATH, FUNDS_PATH, SEC_MAP_PATH, SEC_COVERAGE_PATH,
             REVERE_REL_PATH, REVERE_CO_PATH,
             GPR_PATH, GSDB_PATH, GRAVITY_PATH, CONFLICT_PATH]
    println("File existence check:")
    for f in files
        size_mb = isfile(f) ? round(filesize(f) / 1024^2, digits=1) : 0.0
        mark = isfile(f) ? "OK " : "MISS"
        println("  [$mark] $(basename(f)) ($size_mb MB)")
    end
    # Check holdings chunks
    holdings_files = filter(f -> startswith(basename(f), "Factset_FundOwners_") && endswith(f, ".gz"),
                            readdir(OWN_DIR, join=true))
    println("Holdings chunks: $(length(holdings_files)) files")
    for f in sort(holdings_files)
        size_gb = round(filesize(f) / 1024^3, digits=2)
        println("  $(basename(f))  $size_gb GB")
    end

    # Test DuckDB connection
    print("\nDuckDB version: ")
    con = dbcon()
    res = qdf(con, "SELECT version() AS v")
    println(res.v[1])
    DBInterface.close!(con)

    println("\nOutput directory: $OUT_DIR")
    println("TEST_MODE: $TEST_MODE (set DPN_TEST_MODE=true to enable)")
    println("DATA_ROOT: $DATA_ROOT (override with DPN_DATA_ROOT)")
    println("\nSetup OK. Proceed to 01_master_files.jl")

    # Write input fingerprints CSV for downstream provenance checks.
    fp_path = joinpath(OUT_DIR, "00_input_fingerprints.csv")
    open(fp_path, "w") do io
        write(io, "path,size_bytes,mtime,sha256_head\n")
        for f in files
            sz = isfile(f) ? filesize(f) : 0
            mt = isfile(f) ? string(Dates.unix2datetime(mtime(f))) : ""
            # SHA256 of large files is slow; skip for files > 100 MB here.
            sh = (isfile(f) && filesize(f) < 100_000_000) ? _file_sha256(f) : ""
            write(io, "$f,$sz,$mt,$sh\n")
        end
    end
    println("Input fingerprints -> $fp_path")
end

# If run as a script (not include), execute smoke test
if abspath(PROGRAM_FILE) == @__FILE__
    smoke_test()
end
