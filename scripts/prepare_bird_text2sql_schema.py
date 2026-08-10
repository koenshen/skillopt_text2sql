#!/usr/bin/env python3
"""Convert BIRD SQLite schemas to the host Agent's ``database.json`` format.

Authoritative schema metadata comes from ``train_tables.json``.  Three sample
rows per table are read from the corresponding SQLite database so the existing
``sql_db_schema`` tool can present the same information shape used by the
current BIRD Mini-dev topic.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-json",
        type=Path,
        required=True,
        help="BIRD train_tables.json or dev_tables.json.",
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        required=True,
        help="Directory containing <db_id>/<db_id>.sqlite.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination root for <db_id>/database.json.",
    )
    parser.add_argument("--db-ids", nargs="*", default=None)
    parser.add_argument("--description-prefix", default="BIRD database")
    return parser.parse_args()


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def read_sample_rows(
    connection: sqlite3.Connection,
    table_name: str,
    column_names: list[str],
) -> list[dict[str, Any]]:
    if not column_names:
        return []
    columns_sql = ", ".join(quote_identifier(name) for name in column_names)
    table_sql = quote_identifier(table_name)
    cursor = connection.execute(f"SELECT {columns_sql} FROM {table_sql} LIMIT 3")
    return [
        {name: serialize_value(value) for name, value in zip(column_names, row)}
        for row in cursor.fetchall()
    ]


def build_foreign_keys(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    original_columns = metadata["column_names_original"]
    original_tables = metadata["table_names_original"]
    foreign_keys: list[dict[str, Any]] = []
    for from_index, to_index in metadata.get("foreign_keys", []):
        from_table_index, from_column = original_columns[from_index]
        to_table_index, to_column = original_columns[to_index]
        foreign_keys.append(
            {
                "from": {
                    "table": original_tables[from_table_index],
                    "columns": [from_column],
                },
                "to": {
                    "table": original_tables[to_table_index],
                    "columns": [to_column],
                },
            }
        )
    return foreign_keys


def flatten_primary_keys(values: list[Any]) -> set[int]:
    keys: set[int] = set()
    for value in values:
        if isinstance(value, list):
            keys.update(int(item) for item in value)
        else:
            keys.add(int(value))
    return keys


def convert_database(
    metadata: dict[str, Any], database_root: Path, output_root: Path, description_prefix: str
) -> Path:
    db_id = metadata["db_id"]
    sqlite_path = database_root / db_id / f"{db_id}.sqlite"
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")

    original_tables = metadata["table_names_original"]
    normalized_tables = metadata["table_names"]
    original_columns = metadata["column_names_original"]
    normalized_columns = metadata["column_names"]
    column_types = metadata["column_types"]
    primary_keys = flatten_primary_keys(metadata.get("primary_keys", []))

    columns_by_table: dict[int, list[tuple[int, str, str, str]]] = {
        index: [] for index in range(len(original_tables))
    }
    for column_index, ((table_index, original_name), (_, normalized_name), data_type) in enumerate(
        zip(original_columns, normalized_columns, column_types)
    ):
        if table_index < 0:  # BIRD's synthetic wildcard column
            continue
        columns_by_table[table_index].append(
            (column_index, original_name, normalized_name, data_type)
        )

    tables: list[dict[str, Any]] = []
    connection = sqlite3.connect(str(sqlite_path))
    try:
        physical_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        missing_tables = set(original_tables) - physical_tables
        if missing_tables:
            raise ValueError(f"{db_id} metadata tables missing in SQLite: {sorted(missing_tables)}")

        for table_index, table_name in enumerate(original_tables):
            column_specs = columns_by_table[table_index]
            column_names = [spec[1] for spec in column_specs]
            columns = []
            for column_index, original_name, normalized_name, data_type in column_specs:
                comment = normalized_name if normalized_name != original_name else ""
                columns.append(
                    {
                        "column_name": original_name,
                        "column_comment": comment,
                        "data_type": data_type.upper(),
                        "is_primary_key": column_index in primary_keys,
                        "is_enum": False,
                        "enum_values": None,
                        "enum_comment": None,
                        "enable_entity_retrieval": False,
                    }
                )
            normalized_table = normalized_tables[table_index]
            tables.append(
                {
                    "table_name": table_name,
                    "table_comment": normalized_table if normalized_table != table_name else "",
                    "sample_rows": read_sample_rows(connection, table_name, column_names),
                    "columns": columns,
                }
            )
    finally:
        connection.close()

    payload = {
        "name": db_id,
        "description": f"{description_prefix}: {db_id}",
        "databases": [
            {
                "db_name": db_id,
                "db_comment": f"{description_prefix}: {db_id}",
                "tables": tables,
                "foreign_keys": build_foreign_keys(metadata),
            }
        ],
    }
    output_path = output_root / db_id / "database.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output_path


def main() -> None:
    args = parse_args()
    with args.tables_json.open(encoding="utf-8") as handle:
        metadata_list = json.load(handle)
    selected_ids = set(args.db_ids) if args.db_ids else None
    selected = [
        metadata for metadata in metadata_list
        if selected_ids is None or metadata["db_id"] in selected_ids
    ]
    if selected_ids:
        missing = selected_ids - {metadata["db_id"] for metadata in selected}
        if missing:
            raise ValueError(f"unknown db_id values: {sorted(missing)}")

    print(f"Converting {len(selected)} BIRD databases...")
    for index, metadata in enumerate(selected, 1):
        output_path = convert_database(
            metadata, args.database_root, args.output_dir, args.description_prefix
        )
        print(f"[{index}/{len(selected)}] {metadata['db_id']} -> {output_path}")
    print(f"Done: {len(selected)}/{len(selected)} databases converted")


if __name__ == "__main__":
    main()
