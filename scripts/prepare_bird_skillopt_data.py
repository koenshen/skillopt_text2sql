#!/usr/bin/env python3
"""Materialize BIRD Train/Dev data for the Text-to-SQL SkillOpt adapter.

Only Train and Dev are materialized.  Mini-dev is read solely as an ID
exclusion mask so that the resulting Dev split does not overlap the held-out
Mini-dev/Test split.

Ground-truth SQL is taken from each JSON record's ``SQL`` field.  The script
never joins SQL from a separate line-oriented gold file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_TRAIN_COUNT = 9_428
EXPECTED_FULL_DEV_COUNT = 1_534
EXPECTED_EXCLUDED_MINI_DEV_COUNT = 500
EXPECTED_DEV_COUNT = 1_034
DEFAULT_SEED = 42
DEFAULT_TRAIN_SIZES = (10, 200, 500, 1_000)
DEFAULT_GATE_SIZES = (50, 100, 200)


def parse_args() -> argparse.Namespace:
    # This script lives under <agent-project>/skillopt/scripts/. Keep the host
    # Agent datasets outside SkillOpt and only materialize derived splits in
    # skillopt/data/.
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: skillopt/data/bird_text2sql).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return value


def require_fields(
    records: Iterable[dict[str, Any]],
    fields: tuple[str, ...],
    label: str,
    *,
    nonempty_fields: tuple[str, ...] = (),
) -> None:
    for index, record in enumerate(records):
        missing = [field for field in fields if field not in record]
        if missing:
            raise ValueError(f"{label}[{index}] is missing fields: {missing}")
        for field in nonempty_fields:
            if record[field] is None or (isinstance(record[field], str) and not record[field].strip()):
                raise ValueError(f"{label}[{index}].{field} is empty")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocate_proportionally(
    counts: dict[tuple[str, ...], int],
    size: int,
    *,
    ensure_all_strata: bool = False,
) -> dict[tuple[str, ...], int]:
    """Allocate ``size`` slots across strata using largest remainders."""
    total = sum(counts.values())
    if size < 0 or size > total:
        raise ValueError(f"sample size {size} is outside [0, {total}]")
    if total == 0:
        return {}

    exact = {key: size * count / total for key, count in counts.items()}
    allocation = {key: min(counts[key], int(value)) for key, value in exact.items()}
    remaining = size - sum(allocation.values())
    order = sorted(
        counts,
        key=lambda key: (exact[key] - int(exact[key]), counts[key], key),
        reverse=True,
    )
    for key in order:
        if remaining == 0:
            break
        if allocation[key] < counts[key]:
            allocation[key] += 1
            remaining -= 1
    if remaining:
        raise AssertionError(f"failed to allocate {remaining} stratified slots")

    if ensure_all_strata and size >= len(counts):
        empty_keys = [key for key in sorted(counts) if allocation[key] == 0]
        for empty_key in empty_keys:
            donors = [key for key in counts if allocation[key] > 1]
            if not donors:
                raise AssertionError("cannot guarantee coverage for every stratum")
            donor = max(
                donors,
                key=lambda key: (allocation[key] - exact[key], allocation[key], key),
            )
            allocation[donor] -= 1
            allocation[empty_key] = 1
    return allocation


def build_nested_stratified_order(
    records: list[dict[str, Any]],
    seed: int,
    prefix_sizes: tuple[int, ...],
    *,
    key_fields: tuple[str, ...],
    ensure_all_strata: bool = False,
) -> list[dict[str, Any]]:
    """Build nested, deterministic stratified prefixes of several sizes."""
    sizes = tuple(sorted(set(prefix_sizes)))
    if not sizes or sizes[-1] > len(records):
        raise ValueError(f"invalid prefix sizes {sizes} for {len(records)} records")

    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[tuple(str(record[field]) for field in key_fields)].append(record)

    rng = random.Random(seed)
    for key in sorted(strata):
        rng.shuffle(strata[key])
    counts = {key: len(items) for key, items in strata.items()}

    allocations = {
        size: allocate_proportionally(
            counts,
            size,
            ensure_all_strata=ensure_all_strata,
        )
        for size in sizes
    }
    for previous_size, size in zip(sizes, sizes[1:]):
        for key in counts:
            if allocations[size][key] < allocations[previous_size][key]:
                raise AssertionError(
                    f"non-monotonic nested allocation for {key}: "
                    f"{previous_size}->{size}"
                )

    order: list[dict[str, Any]] = []
    previous = {key: 0 for key in counts}
    for level, size in enumerate(sizes):
        increment: list[dict[str, Any]] = []
        for key in sorted(strata):
            take = allocations[size][key]
            increment.extend(strata[key][previous[key]:take])
            previous[key] = take
        random.Random(seed + 20_000 + level).shuffle(increment)
        order.extend(increment)

    outside = [record for key in sorted(strata) for record in strata[key][previous[key]:]]
    random.Random(seed + 30_000).shuffle(outside)
    order.extend(outside)
    if len(order) != len(records) or len({record["id"] for record in order}) != len(order):
        raise AssertionError("stratified ordering lost or duplicated records")
    return order


def distribution(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    result = {
        "count": len(records),
        "by_database": dict(sorted(Counter(record["db_id"] for record in records).items())),
    }
    if records and all("difficulty" in record for record in records):
        result["by_difficulty"] = dict(
            sorted(Counter(record["difficulty"] for record in records).items())
        )
    return result


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir or repo_root / "skillopt/data/bird_text2sql").resolve()

    train_path = repo_root / "bird_train_datas/train.json"
    full_dev_path = repo_root / "bird_dev_datas/dev.json"
    mini_dev_path = repo_root / "bird_mini_dev_datas/dev.json"
    train = read_json_list(train_path)
    full_dev = read_json_list(full_dev_path)
    mini_dev = read_json_list(mini_dev_path)

    require_fields(
        train,
        ("db_id", "question", "evidence", "SQL"),
        "train",
        nonempty_fields=("db_id", "question", "SQL"),
    )
    require_fields(
        full_dev,
        ("question_id", "db_id", "question", "evidence", "SQL", "difficulty"),
        "full_dev",
        nonempty_fields=("db_id", "question", "SQL", "difficulty"),
    )
    require_fields(mini_dev, ("question_id",), "mini_dev")

    if len(train) != EXPECTED_TRAIN_COUNT:
        raise ValueError(f"expected {EXPECTED_TRAIN_COUNT} Train records, found {len(train)}")
    if len(full_dev) != EXPECTED_FULL_DEV_COUNT:
        raise ValueError(f"expected {EXPECTED_FULL_DEV_COUNT} Full Dev records, found {len(full_dev)}")
    if len(mini_dev) != EXPECTED_EXCLUDED_MINI_DEV_COUNT:
        raise ValueError(
            f"expected {EXPECTED_EXCLUDED_MINI_DEV_COUNT} Mini-dev exclusion IDs, found {len(mini_dev)}"
        )

    full_dev_ids = [record["question_id"] for record in full_dev]
    mini_dev_ids = [record["question_id"] for record in mini_dev]
    if len(set(full_dev_ids)) != len(full_dev_ids):
        raise ValueError("Full Dev contains duplicate question_id values")
    if len(set(mini_dev_ids)) != len(mini_dev_ids):
        raise ValueError("Mini-dev contains duplicate question_id values")
    unknown_ids = set(mini_dev_ids) - set(full_dev_ids)
    if unknown_ids:
        raise ValueError(f"Mini-dev IDs are not a Full Dev subset: {sorted(unknown_ids)[:10]}")

    excluded_ids = set(mini_dev_ids)
    retained_dev_source = [record for record in full_dev if record["question_id"] not in excluded_ids]
    if len(retained_dev_source) != EXPECTED_DEV_COUNT:
        raise ValueError(f"expected {EXPECTED_DEV_COUNT} retained Dev records, found {len(retained_dev_source)}")

    train_public: list[dict[str, Any]] = []
    train_private: list[dict[str, Any]] = []
    for source_index, record in enumerate(train):
        item_id = f"bird_train:{source_index:08d}"
        common = {
            "id": item_id,
            "source_split": "train",
            "source_index": source_index,
            "db_id": record["db_id"],
        }
        train_public.append(
            {
                **common,
                "question": record["question"],
                "evidence": record["evidence"],
            }
        )
        train_private.append({**common, "ground_truth_sql": record["SQL"]})

    dev_public: list[dict[str, Any]] = []
    dev_private: list[dict[str, Any]] = []
    for source_index, record in enumerate(full_dev):
        if record["question_id"] in excluded_ids:
            continue
        item_id = f"bird_dev:{record['question_id']}"
        common = {
            "id": item_id,
            "source_split": "dev",
            "source_index": source_index,
            "question_id": record["question_id"],
            "db_id": record["db_id"],
            "difficulty": record["difficulty"],
        }
        dev_public.append(
            {
                **common,
                "question": record["question"],
                "evidence": record["evidence"],
            }
        )
        dev_private.append({**common, "ground_truth_sql": record["SQL"]})

    train_order = build_nested_stratified_order(
        train_public,
        args.seed,
        DEFAULT_TRAIN_SIZES,
        key_fields=("db_id",),
        ensure_all_strata=True,
    )
    train_order_payload = {
        "seed": args.seed,
        "source": "BIRD Train",
        "stratification": ["db_id"],
        "ensure_all_databases_when_size_allows": True,
        "nested_prefix_sizes": list(DEFAULT_TRAIN_SIZES),
        "ordered_items": [
            {
                "id": record["id"],
                "source_index": record["source_index"],
                "db_id": record["db_id"],
            }
            for record in train_order
        ],
    }

    gate_order = build_nested_stratified_order(
        dev_public,
        args.seed,
        DEFAULT_GATE_SIZES,
        key_fields=("db_id", "difficulty"),
    )
    gate_order_payload = {
        "seed": args.seed,
        "source": "retained Full Dev after excluding all Mini-dev question_id values",
        "stratification": ["db_id", "difficulty"],
        "nested_prefix_sizes": list(DEFAULT_GATE_SIZES),
        "ordered_items": [
            {
                "id": record["id"],
                "question_id": record["question_id"],
                "db_id": record["db_id"],
                "difficulty": record["difficulty"],
            }
            for record in gate_order
        ],
    }

    paths = {
        "train_public": output_dir / "public/train.jsonl",
        "dev_public": output_dir / "public/dev.jsonl",
        "train_private": output_dir / "private/train_gold.jsonl",
        "dev_private": output_dir / "private/dev_gold.jsonl",
        "train_order": output_dir / "selections/train_order.json",
        "gate_order": output_dir / "selections/dev_gate_order.json",
        "split_report": output_dir / "reports/split_report.json",
    }
    write_jsonl(paths["train_public"], train_public)
    write_jsonl(paths["dev_public"], dev_public)
    write_jsonl(paths["train_private"], train_private)
    write_jsonl(paths["dev_private"], dev_private)
    write_json(paths["train_order"], train_order_payload)
    write_json(paths["gate_order"], gate_order_payload)

    split_report = {
        "counts": {
            "train": len(train_public),
            "full_dev": len(full_dev),
            "mini_dev_ids_used_only_for_exclusion": len(excluded_ids),
            "retained_dev": len(dev_public),
        },
        "ground_truth_policy": {
            "authoritative_field": "SQL in the same source JSON record",
            "line_oriented_gold_files_used": False,
            "precomputed_execution_results": False,
        },
        "gate_profiles": {
            str(size): distribution(gate_order[:size]) for size in DEFAULT_GATE_SIZES
        },
        "train_profiles": {
            str(size): distribution(train_order[:size]) for size in DEFAULT_TRAIN_SIZES
        },
        "full_train_distribution": distribution(train_public),
        "full_dev_distribution": distribution(dev_public),
    }
    write_json(paths["split_report"], split_report)

    manifest = {
        "format_version": 2,
        "seed": args.seed,
        "sources": {
            "train_json": str(train_path.relative_to(repo_root)),
            "full_dev_json": str(full_dev_path.relative_to(repo_root)),
            "mini_dev_exclusion_json": str(mini_dev_path.relative_to(repo_root)),
            "train_database_root": "bird_train_datas/train_databases",
            "dev_database_root": "bird_dev_datas/dev_databases",
        },
        "source_sha256": {
            "train_json": sha256_file(train_path),
            "full_dev_json": sha256_file(full_dev_path),
            "mini_dev_exclusion_json": sha256_file(mini_dev_path),
        },
        "outputs": {name: str(path.relative_to(output_dir)) for name, path in paths.items()},
        "counts": {"train": len(train_public), "dev": len(dev_public)},
        "gate": {
            "config_key": "evaluation.sel_env_num",
            "test_run_size": 50,
            "pilot_run_size": 100,
            "full_run_size": 200,
            "full_dev_value": 0,
            "fixed_order_file": "selections/dev_gate_order.json",
        },
        "train_selection": {
            "config_key": "train.train_size",
            "test_run_size": 10,
            "epoch2_pilot_run_size": 200,
            "pilot_run_size": 500,
            "default_run_size": 1000,
            "fixed_order_file": "selections/train_order.json",
        },
        "test_materialized": False,
    }
    write_json(output_dir / "manifest.json", manifest)

    print(f"Wrote BIRD SkillOpt Train/Dev data to {output_dir}")
    print(f"  Train: {len(train_public)}")
    print(f"  Dev:   {len(dev_public)}")
    for size in DEFAULT_TRAIN_SIZES:
        profile = split_report["train_profiles"][str(size)]
        print(f"  Train {size}: databases={len(profile['by_database'])}")
    for size in DEFAULT_GATE_SIZES:
        profile = split_report["gate_profiles"][str(size)]
        print(f"  Gate {size}: {profile['by_difficulty']}")


if __name__ == "__main__":
    main()
