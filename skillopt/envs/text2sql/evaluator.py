"""Official BIRD execution-accuracy logic adapted for single-item rewards.

The execution and comparison functions below are copied from
``bird_evaluate/src/evaluation_utils.py`` and
``bird_evaluate/src/evaluation_v2.py``.  The only adaptation is the
``evaluate_one`` entry point: SkillOpt already has the predicted SQL, gold SQL,
and database path in memory, so it does not package them through benchmark
files before calling the same execution kernel.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from func_timeout import FunctionTimedOut, func_timeout


def connect_db(sql_dialect: str, db_path: str):
    """Copied from the official BIRD evaluator."""
    if sql_dialect == "SQLite":
        conn = sqlite3.connect(db_path)
    elif sql_dialect == "MySQL":
        import pymysql

        conn = pymysql.connect(
            host="localhost",
            user="root",
            password="",
            database="BIRD",
            unix_socket="/var/run/mysqld/mysqld.sock",
        )
    elif sql_dialect == "PostgreSQL":
        import psycopg2

        conn = psycopg2.connect(
            "dbname=bird user=postgres host=localhost password= port=5432"
        )
    else:
        raise ValueError(f"Unsupported SQL dialect: {sql_dialect}")
    return conn


def execute_sql(
    predicted_sql: str,
    ground_truth: str,
    db_path: str,
    sql_dialect: str,
    calculate_func,
):
    """Copied from the official BIRD evaluator without semantic changes."""
    conn = connect_db(sql_dialect, db_path)
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    conn.close()
    res = calculate_func(predicted_res, ground_truth_res)
    return res, predicted_res, ground_truth_res


def calculate_ex(predicted_res: list[tuple], ground_truth_res: list[tuple]) -> int:
    """Official BIRD execution-accuracy comparison."""
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def execute_model(
    predicted_sql: str,
    ground_truth: str,
    db_place: str,
    idx: int,
    meta_time_out: float,
    sql_dialect: str,
) -> dict[str, Any]:
    """Official per-query timeout/error wrapper."""
    predicted_res, ground_truth_res = None, None
    status = "correct"
    error = ""
    try:
        res, predicted_res, ground_truth_res = func_timeout(
            meta_time_out,
            execute_sql,
            args=(predicted_sql, ground_truth, db_place, sql_dialect, calculate_ex),
        )
        if not res:
            status = "result_mismatch"
    except FunctionTimedOut:
        res = 0
        status = "timeout"
    except Exception as exc:  # official evaluator maps every exception to res=0
        res = 0
        status = "execution_error"
        error = f"{type(exc).__name__}: {exc}"
    return {
        "sql_idx": idx,
        "res": res,
        "predicted_res": predicted_res,
        "ground_truth_res": ground_truth_res,
        "predicted_sql": predicted_sql,
        "ground_truth": ground_truth,
        "status": status,
        "error": error,
    }


def evaluate_one(
    predicted_sql: str,
    ground_truth_sql: str,
    db_path: str | Path,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Evaluate one prediction with the official BIRD EX implementation."""
    path = Path(db_path)
    if not predicted_sql or not predicted_sql.strip():
        return {
            "sql_idx": 0,
            "res": 0,
            "predicted_res": None,
            "ground_truth_res": None,
            "predicted_sql": predicted_sql,
            "ground_truth": ground_truth_sql,
            "status": "empty_prediction",
            "error": "",
        }
    if not path.is_file():
        return {
            "sql_idx": 0,
            "res": 0,
            "predicted_res": None,
            "ground_truth_res": None,
            "predicted_sql": predicted_sql,
            "ground_truth": ground_truth_sql,
            "status": "missing_database",
            "error": f"SQLite database not found: {path}",
        }
    return execute_model(
        predicted_sql=predicted_sql,
        ground_truth=ground_truth_sql,
        db_place=str(path),
        idx=0,
        meta_time_out=float(timeout),
        sql_dialect="SQLite",
    )
