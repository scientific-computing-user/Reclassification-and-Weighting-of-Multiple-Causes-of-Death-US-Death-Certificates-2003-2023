#!/usr/bin/env python3
"""
Validate key numerical claims from the paper against the local all_I snapshot.

Default mode uses `sel_code` as the Entity-axis comparison field.
Use `--entity-column ent_ucod` to evaluate an alternate/sensitivity path.
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Any


DL_NAME = {
    "B": "Circulatory",
    "C": "Cancer",
    "N": "Other Natural",
    "R": "Respiratory",
    "E": "Endocrine",
    "D": "Digestive",
    "V": "COVID-19",
    "P": "Drug Poisoning",
    "T": "Transport",
    "S": "Suicide",
    "A": "Alcohol-related",
    "X": "Other External",
    "F": "Falls",
    "H": "Homicide",
    "U": "Unknown",
}


def pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_lookup(path: str) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if "ICD10" not in reader.fieldnames or "DL" not in reader.fieldnames:
            raise ValueError("Lookup CSV must contain ICD10 and DL columns")
        for row in reader:
            lookup[row["ICD10"]] = row["DL"]
    return lookup


def required_index(header_map: Dict[str, int], column: str) -> int:
    if column not in header_map:
        raise ValueError(f"Required column not found in all_I file: {column}")
    return header_map[column]


def scan_metrics(
    all_i_path: str,
    lookup: Dict[str, str],
    entity_column: str,
    progress_every: int,
) -> Dict[str, Any]:
    stats = Counter()
    period = {"pre": Counter(), "pan": Counter()}
    rec_cat = Counter()
    ent_cat = Counter()
    transitions = Counter()

    start = time.time()
    with open(all_i_path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        header_map = {name: index for index, name in enumerate(header)}

        idx_rads = required_index(header_map, "rADS")
        idx_ruds = required_index(header_map, "rUDS")
        idx_eads = required_index(header_map, "eADS")
        idx_euds = required_index(header_map, "eUDS")
        idx_year = required_index(header_map, "year")
        idx_ucod = required_index(header_map, "ucod")
        idx_dup1 = required_index(header_map, "dup1")
        idx_entity = required_index(header_map, entity_column)

        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue

            stats["total_deaths"] += 1

            rads = row[idx_rads]
            ruds = row[idx_ruds]
            eads = row[idx_eads]
            euds = row[idx_euds]
            stats["total_rads_chars"] += len(rads)
            stats["total_ruds_chars"] += len(ruds)
            stats["total_eads_chars"] += len(eads)
            stats["total_euds_chars"] += len(euds)

            if len(ruds) == 1:
                stats["single_ruds"] += 1
            if len(euds) == 1:
                stats["single_euds"] += 1

            try:
                if int(row[idx_dup1]) > 1:
                    stats["dup1_gt1"] += 1
            except ValueError:
                pass

            ruc = row[idx_ucod]
            euc = row[idx_entity]

            if ruc and euc and ruc == euc:
                stats["icd_concord"] += 1

            rdl = lookup.get(ruc, "")
            edl = lookup.get(euc, "")

            if rdl:
                rec_cat[rdl] += 1
            if edl:
                ent_cat[edl] += 1

            if rdl and edl:
                transition_key = f"{edl}->{rdl}"
                transitions[transition_key] += 1
                if rdl == edl:
                    stats["cat_concord"] += 1

            if ruc == "U071" and euc != "U071":
                stats["covid_promotions"] += 1
                if euc == "J189":
                    stats["covid_promotions_from_J189"] += 1
            if ruc != "U071" and euc == "U071":
                stats["covid_demotions"] += 1

            try:
                year = int(row[idx_year])
                bucket = "pre" if year <= 2019 else "pan"
                period[bucket]["total"] += 1
                if ruc and euc and ruc == euc:
                    period[bucket]["icd_concord"] += 1
                if rdl and edl and rdl == edl:
                    period[bucket]["cat_concord"] += 1
            except ValueError:
                pass

            if progress_every > 0 and row_number % progress_every == 0:
                elapsed = time.time() - start
                rate = row_number / elapsed if elapsed > 0 else 0.0
                print(
                    f"[progress] processed {row_number:,} rows in {elapsed:.1f}s ({rate:,.0f} rows/s)",
                    flush=True,
                )

    elapsed_seconds = time.time() - start
    total = stats["total_deaths"]
    pre_total = period["pre"]["total"]
    pan_total = period["pan"]["total"]

    category_change_pct: Dict[str, float] = {}
    for letter, entity_count in ent_cat.items():
        record_count = rec_cat.get(letter, 0)
        if entity_count > 0:
            category_change_pct[letter] = 100.0 * (record_count - entity_count) / entity_count

    promotion_count = stats["covid_promotions"]
    demotion_count = stats["covid_demotions"]
    promotion_ratio = (promotion_count / demotion_count) if demotion_count else float("inf")

    metrics = {
        "runtime_seconds": elapsed_seconds,
        "entity_column": entity_column,
        "total_deaths": total,
        "single_ruds_count": stats["single_ruds"],
        "single_ruds_pct": pct(stats["single_ruds"], total),
        "single_euds_count": stats["single_euds"],
        "single_euds_pct": pct(stats["single_euds"], total),
        "dup1_gt1_count": stats["dup1_gt1"],
        "dup1_gt1_pct": pct(stats["dup1_gt1"], total),
        "extra_broad_causes_record": stats["total_ruds_chars"] - total,
        "extra_broad_causes_entity": stats["total_euds_chars"] - total,
        "extra_icd_causes_record": stats["total_rads_chars"] - total,
        "extra_icd_causes_entity": stats["total_eads_chars"] - total,
        "concordance_icd_count": stats["icd_concord"],
        "concordance_icd_pct": pct(stats["icd_concord"], total),
        "concordance_category_count": stats["cat_concord"],
        "concordance_category_pct": pct(stats["cat_concord"], total),
        "period_pre_total": pre_total,
        "period_pre_icd_concord_pct": pct(period["pre"]["icd_concord"], pre_total),
        "period_pre_category_concord_pct": pct(period["pre"]["cat_concord"], pre_total),
        "period_pan_total": pan_total,
        "period_pan_icd_concord_pct": pct(period["pan"]["icd_concord"], pan_total),
        "period_pan_category_concord_pct": pct(period["pan"]["cat_concord"], pan_total),
        "covid_promotions": promotion_count,
        "covid_demotions": demotion_count,
        "covid_promotion_to_demotion_ratio": promotion_ratio,
        "covid_promotions_from_J189": stats["covid_promotions_from_J189"],
        "covid_promotions_from_J189_pct": pct(stats["covid_promotions_from_J189"], promotion_count),
        "record_category_counts": dict(rec_cat),
        "entity_category_counts": dict(ent_cat),
        "category_change_pct": category_change_pct,
        "transition_counts": dict(transitions),
    }
    return metrics


def build_claim_checks(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    transitions = metrics["transition_counts"]
    changes = metrics["category_change_pct"]

    checks_spec = [
        ("Total deaths", 56986831, metrics["total_deaths"], 0),
        ("Part-I last-line multiple ICD count", 7749865, metrics["dup1_gt1_count"], 0),
        ("Part-I last-line multiple ICD pct", 13.6, metrics["dup1_gt1_pct"], 0.2),
        ("Category concordance pct", 84.8, metrics["concordance_category_pct"], 1.0),
        ("ICD concordance pct", 68.9, metrics["concordance_icd_pct"], 0.6),
        ("Pre-pandemic category concordance pct", 85.7, metrics["period_pre_category_concord_pct"], 1.0),
        ("Pre-pandemic ICD concordance pct", 70.1, metrics["period_pre_icd_concord_pct"], 0.8),
        ("Pandemic category concordance pct", 81.6, metrics["period_pan_category_concord_pct"], 1.0),
        ("Pandemic ICD concordance pct", 65.0, metrics["period_pan_icd_concord_pct"], 0.8),
        ("Category change COVID-19 pct", 92.0, changes.get("V"), 3.0),
        ("Category change Falls pct", 69.0, changes.get("F"), 3.0),
        ("Category change Transport pct", 44.0, changes.get("T"), 3.0),
        ("Category change Suicide pct", 25.0, changes.get("S"), 3.0),
        ("Category change Homicide pct", 30.0, changes.get("H"), 3.0),
        ("Category change Endocrine pct", 16.0, changes.get("E"), 3.0),
        ("Category change Cancer pct", 12.0, changes.get("C"), 3.0),
        ("Category change Other Natural pct", -14.0, changes.get("N"), 3.0),
        ("Category change Respiratory pct", -11.0, changes.get("R"), 3.0),
        ("Category change Other External pct", -54.0, changes.get("X"), 6.0),
        ("Transition R->V count", 288936, transitions.get("R->V", 0), 5000),
        ("Transition N->V count", 119181, transitions.get("N->V", 0), 5000),
        ("Transition X->T count", 266638, transitions.get("X->T", 0), 5000),
        ("Transition X->F count", 218625, transitions.get("X->F", 0), 5000),
        ("Transition X->S count", 139167, transitions.get("X->S", 0), 5000),
        ("Transition A->P count", 87860, transitions.get("A->P", 0), 5000),
        ("COVID promotions count", 502461, metrics["covid_promotions"], 5000),
        ("COVID demotions count", 20765, metrics["covid_demotions"], 1000),
        (
            "COVID promotion/demotion ratio",
            24.0,
            metrics["covid_promotion_to_demotion_ratio"],
            2.0,
        ),
        ("J189 share of promotions pct", 43.8, metrics["covid_promotions_from_J189_pct"], 2.0),
    ]

    checks: List[Dict[str, Any]] = []
    for label, expected, observed, tolerance in checks_spec:
        if observed is None:
            checks.append(
                {
                    "label": label,
                    "expected": expected,
                    "observed": None,
                    "tolerance": tolerance,
                    "difference": None,
                    "pass": False,
                }
            )
            continue

        difference = observed - expected
        check_pass = abs(difference) <= tolerance
        checks.append(
            {
                "label": label,
                "expected": expected,
                "observed": observed,
                "tolerance": tolerance,
                "difference": difference,
                "pass": check_pass,
            }
        )
    return checks


def format_number(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown_report(path: str, metrics: Dict[str, Any], checks: List[Dict[str, Any]]) -> None:
    passed = sum(1 for item in checks if item["pass"])
    total = len(checks)
    failed = total - passed

    lines: List[str] = []
    lines.append("# Claim Validation Report")
    lines.append("")
    lines.append(f"- Entity comparison column: `{metrics['entity_column']}`")
    lines.append(f"- Total deaths scanned: `{metrics['total_deaths']:,}`")
    lines.append(f"- Runtime (seconds): `{metrics['runtime_seconds']:.1f}`")
    lines.append(f"- Claim checks passed: `{passed}/{total}`")
    lines.append(f"- Claim checks failed: `{failed}`")
    lines.append("")
    lines.append("## Core Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Category concordance (%) | {metrics['concordance_category_pct']:.3f} |")
    lines.append(f"| ICD concordance (%) | {metrics['concordance_icd_pct']:.3f} |")
    lines.append(f"| Pre-pandemic category concordance (%) | {metrics['period_pre_category_concord_pct']:.3f} |")
    lines.append(f"| Pre-pandemic ICD concordance (%) | {metrics['period_pre_icd_concord_pct']:.3f} |")
    lines.append(f"| Pandemic category concordance (%) | {metrics['period_pan_category_concord_pct']:.3f} |")
    lines.append(f"| Pandemic ICD concordance (%) | {metrics['period_pan_icd_concord_pct']:.3f} |")
    lines.append(f"| COVID promotions | {metrics['covid_promotions']:,} |")
    lines.append(f"| COVID demotions | {metrics['covid_demotions']:,} |")
    lines.append(f"| J189 share of promotions (%) | {metrics['covid_promotions_from_J189_pct']:.3f} |")
    lines.append("")
    lines.append("## Paper Claim Checks")
    lines.append("")
    lines.append("| Claim | Expected | Observed | Diff | Tol | Status |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for item in checks:
        status = "PASS" if item["pass"] else "FAIL"
        lines.append(
            "| {label} | {expected} | {observed} | {diff} | {tol} | {status} |".format(
                label=item["label"],
                expected=format_number(item["expected"]),
                observed=format_number(item["observed"]),
                diff=format_number(item["difference"]),
                tol=format_number(item["tolerance"]),
                status=status,
            )
        )
    lines.append("")
    lines.append("## Category Count Changes (Record vs Entity)")
    lines.append("")
    lines.append("| DL | Category | Entity | Record | Change (%) |")
    lines.append("|---|---|---:|---:|---:|")
    entity_counts = metrics["entity_category_counts"]
    record_counts = metrics["record_category_counts"]
    change_pct = metrics["category_change_pct"]
    for letter in ["V", "F", "T", "S", "H", "E", "C", "B", "D", "A", "P", "X", "N", "R", "U"]:
        if letter not in entity_counts and letter not in record_counts:
            continue
        entity_value = entity_counts.get(letter, 0)
        record_value = record_counts.get(letter, 0)
        change_value = change_pct.get(letter)
        change_str = f"{change_value:.3f}" if change_value is not None else "NA"
        lines.append(
            f"| {letter} | {DL_NAME.get(letter, letter)} | {entity_value:,} | {record_value:,} | {change_str} |"
        )

    ensure_parent_dir(path)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paper claims against all_I data")
    parser.add_argument(
        "--all-i",
        default="processed2_GH24/Dup==A/all_I=converted_with_original.csv",
        help="Path to all_I=converted_with_original.csv",
    )
    parser.add_argument(
        "--lookup",
        default="processed2_GH24/icd10_to_DL_lookup_v4.csv",
        help="Path to ICD10->DL lookup CSV",
    )
    parser.add_argument(
        "--entity-column",
        default="sel_code",
        choices=["sel_code", "ent_ucod", "ent_ucod2"],
        help="Entity column used for comparisons",
    )
    parser.add_argument(
        "--output-json",
        default="results/validation/claim_metrics.json",
        help="Output JSON metrics path",
    )
    parser.add_argument(
        "--output-md",
        default="results/validation/claim_check.md",
        help="Output Markdown report path",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000_000,
        help="Print progress every N rows (0 disables progress logs)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any claim check fails",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    lookup = load_lookup(args.lookup)
    metrics = scan_metrics(
        all_i_path=args.all_i,
        lookup=lookup,
        entity_column=args.entity_column,
        progress_every=args.progress_every,
    )
    checks = build_claim_checks(metrics)

    payload = {
        "metrics": metrics,
        "checks": checks,
    }

    ensure_parent_dir(args.output_json)
    with open(args.output_json, "w") as handle:
        json.dump(payload, handle, indent=2)

    write_markdown_report(args.output_md, metrics, checks)

    passed = sum(1 for item in checks if item["pass"])
    total = len(checks)
    failed = total - passed
    print(f"[done] wrote JSON: {args.output_json}")
    print(f"[done] wrote Markdown: {args.output_md}")
    print(f"[summary] checks passed: {passed}/{total}")
    if failed > 0:
        print(f"[summary] checks failed: {failed}")

    if args.strict and failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
