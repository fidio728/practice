# 03a_decompress_to_parquet.jl
# One-time conversion: gz holdings chunks -> raw parquet cache.
#
# Why:
#   * gz decompression is single-threaded zlib and dominates 03 runtime.
#   * Subsequent runs of 03 (re-filter, re-schema) re-pay that ~25-min cost.
#   * Parquet is column-pruned + already typed + already compressed.
#   * After this script, 03 reads parquet and finishes in ~1-2 min.
#
# Output location: RAW_PARQUET_DIR (set in 00_setup.jl, OFF the OneDrive
# path to avoid syncing ~30 GB to the cloud).
#
# Behavior:
#   * SKIP_EXISTING = true (default): skip chunks whose parquet already exists.
#     Lets you resume after disk-full or Ctrl-C.
#   * Prints size after each chunk so you can abort if disk fills up.
#   * Keeps ALL columns (full schema). Slim derivatives are 03's job.

include("00_setup.jl")

const SKIP_EXISTING = true

println("=" ^ 60)
println("Raw-parquet cache builder")
println("  source:  $OWN_DIR")
println("  target:  $RAW_PARQUET_DIR")
println("  skip-if-exists: $SKIP_EXISTING")
println("=" ^ 60)

# Sanity: check the target drive has headroom
try
    df = Base.diskstat(RAW_PARQUET_DIR)
    free_gb = round(df.available / 1024^3, digits=1)
    println("  free space on target drive: $free_gb GB")
catch
    println("  (could not query target-drive free space)")
end
println()

# List source gz files
gz_files = sort(filter(f -> startswith(basename(f), "Factset_FundOwners_") &&
                            endswith(f, ".gz"),
                       readdir(OWN_DIR, join=true)))

println("Found $(length(gz_files)) source chunks to convert.\n")

con = dbcon(memory_gb=6, threads=4)

total_gz_gb = 0.0
total_pq_gb = 0.0
t_start = time()

for (i, gz) in enumerate(gz_files)
    global total_gz_gb, total_pq_gb   # script-scope accumulators
    base    = replace(basename(gz), ".gz" => "")
    out_pq  = joinpath(RAW_PARQUET_DIR, "$base.parquet")
    gz_gb   = filesize(gz) / 1024^3
    total_gz_gb += gz_gb

    @printf("[%d/%d] %-40s  gz=%5.2f GB  ", i, length(gz_files), basename(gz), gz_gb)

    if SKIP_EXISTING && isfile(out_pq)
        pq_gb = filesize(out_pq) / 1024^3
        total_pq_gb += pq_gb
        @printf("EXISTS  parquet=%5.2f GB  (skip)\n", pq_gb)
        continue
    end

    # Atomic write: stream to *.parquet.tmp, then rename to *.parquet on
    # success. Prevents a half-written .parquet from being mistaken for a
    # completed conversion on the next SKIP_EXISTING run (Ctrl-C, OOM, etc).
    tmp_pq = out_pq * ".tmp"
    # Clean any stale .tmp from a previous interrupted run
    isfile(tmp_pq) && rm(tmp_pq; force=true)

    t0 = time()
    try
        # Stream-read gz, write parquet with zstd. Use sample_size=-1 because
        # the ETL needs the FULL accurate schema, not a sampled one (or risk
        # mistyped rare-null columns).
        DBInterface.execute(con, """
            COPY (
                SELECT * FROM read_csv_auto(
                    '$(replace(gz, "\\" => "/"))',
                    compression='gzip',
                    sample_size=-1
                )
            ) TO '$(replace(tmp_pq, "\\" => "/"))' (FORMAT 'parquet', COMPRESSION 'zstd')
        """)
        # Atomic rename on same filesystem
        mv(tmp_pq, out_pq; force=true)
    catch e
        # On failure, drop the partial .tmp so the next run doesn't see it
        isfile(tmp_pq) && rm(tmp_pq; force=true)
        rethrow(e)
    end
    dt = time() - t0
    pq_gb = filesize(out_pq) / 1024^3
    total_pq_gb += pq_gb

    @printf("DONE  parquet=%5.2f GB  ratio=%.2fx  %4.1f min\n",
            pq_gb, pq_gb / gz_gb, dt / 60)
end

DBInterface.close!(con)

t_total = time() - t_start

println()
println("=" ^ 60)
println("DONE")
@printf("  total wallclock:    %.1f min\n", t_total / 60)
@printf("  total gz size:      %.2f GB\n", total_gz_gb)
@printf("  total parquet size: %.2f GB\n", total_pq_gb)
@printf("  parquet/gz ratio:   %.2fx\n", total_pq_gb / total_gz_gb)
println()
println("Next: modify 03_eom_etl.jl to read from RAW_PARQUET_DIR instead of gz")
println("(if not already done). Subsequent 03 runs should take ~1-2 min.")
