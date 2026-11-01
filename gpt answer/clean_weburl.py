import argparse
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


def find_default_input() -> Path:
    here = Path(__file__).resolve().parent
    for parent in here.parents:
        candidate = parent / "bot_paper" / "compustat_2025.csv"
        if candidate.exists():
            return candidate
    return here / "compustat_2025.csv"


def clean_one_url(value: object) -> dict[str, object]:
    if pd.isna(value):
        return {
            "host_clean": pd.NA,
            "weburl_missing": True,
            "url_parse_error": False,
            "host_source_scheme": "missing_scheme",
        }

    raw = str(value).strip()
    if raw == "":
        return {
            "host_clean": pd.NA,
            "weburl_missing": True,
            "url_parse_error": False,
            "host_source_scheme": "missing_scheme",
        }

    parse_target = raw
    if raw.lower().startswith("https://"):
        source_scheme = "https"
    elif raw.lower().startswith("http://"):
        source_scheme = "http"
    else:
        source_scheme = "missing_scheme"
        parse_target = "https://" + raw

    try:
        parsed = urlparse(parse_target)
        host = (parsed.hostname or "").rstrip(".").lower()
    except ValueError:
        host = ""

    if not host or any(char.isspace() for char in host):
        return {
            "host_clean": pd.NA,
            "weburl_missing": False,
            "url_parse_error": True,
            "host_source_scheme": source_scheme,
        }

    return {
        "host_clean": host,
        "weburl_missing": False,
        "url_parse_error": False,
        "host_source_scheme": source_scheme,
    }


def clean_compustat(input_path: Path | str, output_path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(input_path, dtype={"gvkey": str, "cik": str, "cusip": str, "tic": str})
    cleaned = df["weburl"].apply(clean_one_url).apply(pd.Series)

    result = df.copy()
    result["host_clean"] = cleaned["host_clean"]
    result["weburl_missing"] = cleaned["weburl_missing"].astype(bool)
    result["url_parse_error"] = cleaned["url_parse_error"].astype(bool)
    result["host_source_scheme"] = cleaned["host_source_scheme"]

    host_counts = (
        result.loc[~result["weburl_missing"] & ~result["url_parse_error"], "host_clean"]
        .value_counts(dropna=True)
        .to_dict()
    )

    result["host_duplicate_count"] = result["host_clean"].map(host_counts).fillna(0).astype(int)
    result["shared_host_flag"] = result["host_duplicate_count"].gt(1)

    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Compustat web URLs into canonical hosts.")
    parser.add_argument("--input", type=Path, default=find_default_input())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "compustat_2025_clean.csv",
    )
    args = parser.parse_args()

    clean_compustat(args.input, args.output)


if __name__ == "__main__":
    main()
