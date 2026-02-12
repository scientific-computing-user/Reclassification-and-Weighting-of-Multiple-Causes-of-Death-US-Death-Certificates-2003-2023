#!/usr/bin/env python3
"""
Build compact JSON assets for the interactive docs website.
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple


SCHEME_NAME = {
    "1": "W0 (Unweighted)",
    "2": "W1 (Half + Share)",
    "3": "W2 (Equal UDS)",
    "4": "W2A (Equal ADS)",
}

LETTER_NAME = {
    "A": "Alcohol-related",
    "B": "Circulatory",
    "C": "Cancer",
    "D": "Digestive",
    "E": "Endocrine",
    "F": "Falls",
    "H": "Homicide",
    "N": "Other Natural",
    "P": "Drug Poisoning",
    "R": "Respiratory",
    "S": "Suicide",
    "T": "Transport",
    "U": "Unknown",
    "V": "COVID-19",
    "X": "Other External",
}


def aggregate_rows(
    path: Path,
    group_fields: Iterable[str],
    sum_field: str = "weight",
    filters: Dict[str, set] = None,
) -> Dict[Tuple[str, ...], float]:
    filters = filters or {}
    grouped: Dict[Tuple[str, ...], float] = defaultdict(float)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            should_skip = False
            for key, allowed in filters.items():
                if row.get(key) not in allowed:
                    should_skip = True
                    break
            if should_skip:
                continue
            key = tuple(row[field] for field in group_fields)
            grouped[key] += float(row[sum_field])
    return grouped


def build_output(
    year_csv: Path,
    month_csv: Path,
    overall_csv: Path,
    claim_json: Path,
    output_json: Path,
) -> None:
    yearly = aggregate_rows(
        path=year_csv,
        group_fields=["scheme", "bucket", "year", "letter"],
        sum_field="weight",
    )

    monthly = aggregate_rows(
        path=month_csv,
        group_fields=["scheme", "bucket", "month_idx", "letter"],
        sum_field="weight",
        filters={"bucket": {"ALL", "UNDER65", "PLUS65"}},
    )

    overall = aggregate_rows(
        path=overall_csv,
        group_fields=["scheme", "bucket", "letter"],
        sum_field="weight",
    )

    with claim_json.open() as handle:
        claim_payload = json.load(handle)

    metrics = claim_payload["metrics"]
    checks = claim_payload["checks"]

    yearly_records = [
        {
            "scheme": scheme,
            "bucket": bucket,
            "year": int(year),
            "letter": letter,
            "weight": round(weight, 6),
        }
        for (scheme, bucket, year, letter), weight in sorted(yearly.items())
    ]

    monthly_records = [
        {
            "scheme": scheme,
            "bucket": bucket,
            "month_idx": int(month_idx),
            "year": 2003 + int(month_idx) // 12,
            "month": 1 + int(month_idx) % 12,
            "letter": letter,
            "weight": round(weight, 6),
        }
        for (scheme, bucket, month_idx, letter), weight in sorted(monthly.items())
    ]

    overall_records = [
        {
            "scheme": scheme,
            "bucket": bucket,
            "letter": letter,
            "weight": round(weight, 6),
        }
        for (scheme, bucket, letter), weight in sorted(overall.items())
    ]

    checks_records = [
        {
            "label": item["label"],
            "expected": item["expected"],
            "observed": item["observed"],
            "tolerance": item["tolerance"],
            "difference": item["difference"],
            "pass": bool(item["pass"]),
        }
        for item in checks
    ]

    transition_records = [
        {
            "from": edge.split("->")[0],
            "to": edge.split("->")[1],
            "count": int(count),
        }
        for edge, count in sorted(
            metrics["transition_counts"].items(),
            key=lambda pair: pair[1],
            reverse=True,
        )
    ]

    payload = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_files": {
                "year_csv": str(year_csv),
                "month_csv": str(month_csv),
                "overall_csv": str(overall_csv),
                "claim_json": str(claim_json),
            },
            "monthly_buckets_included": ["ALL", "UNDER65", "PLUS65"],
        },
        "labels": {
            "scheme_name": SCHEME_NAME,
            "letter_name": LETTER_NAME,
        },
        "claim_metrics": {
            "summary": {
                "total_deaths": metrics["total_deaths"],
                "concordance_category_pct": metrics["concordance_category_pct"],
                "concordance_icd_pct": metrics["concordance_icd_pct"],
                "period_pre_category_concord_pct": metrics["period_pre_category_concord_pct"],
                "period_pan_category_concord_pct": metrics["period_pan_category_concord_pct"],
                "covid_promotions": metrics["covid_promotions"],
                "covid_demotions": metrics["covid_demotions"],
                "covid_promotion_to_demotion_ratio": metrics["covid_promotion_to_demotion_ratio"],
                "covid_promotions_from_J189_pct": metrics["covid_promotions_from_J189_pct"],
                "fallback_record_axis": metrics.get("fallback_record_axis", 0),
                "fallback_entity_axis": metrics.get("fallback_entity_axis", 0),
                "external_prefix_fallback": metrics.get("external_prefix_fallback", ""),
            },
            "category_change_pct": metrics["category_change_pct"],
            "record_category_counts": metrics["record_category_counts"],
            "entity_category_counts": metrics["entity_category_counts"],
            "checks": checks_records,
            "transitions": transition_records,
        },
        "series": {
            "yearly": yearly_records,
            "monthly": monthly_records,
            "overall": overall_records,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump(payload, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build docs site dataset JSON")
    parser.add_argument("--year-csv", default="data/output_agg/year_agg.csv")
    parser.add_argument("--month-csv", default="data/output_agg/month_idx_agg.csv")
    parser.add_argument("--overall-csv", default="data/output_agg/overall_agg.csv")
    parser.add_argument("--claim-json", default="results/validation/claim_metrics.json")
    parser.add_argument("--output-json", default="docs/assets/site_data.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_output(
        year_csv=Path(args.year_csv),
        month_csv=Path(args.month_csv),
        overall_csv=Path(args.overall_csv),
        claim_json=Path(args.claim_json),
        output_json=Path(args.output_json),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
