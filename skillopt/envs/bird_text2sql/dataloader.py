"""BIRD Text-to-SQL data loader for SkillOpt.

Loads pre-split prompt.jsonl files from train/val/test directories.
Each line in prompt.jsonl contains:
  question_id, db_id, question, evidence, full_question, SQL, schema, prompt, difficulty
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from skillopt.datasets.base import SplitDataLoader


def _normalize_item(raw: dict) -> dict:
    """Normalize one BIRD prompt entry into the dict shape SkillOpt expects."""
    return {
        "id": str(raw.get("question_id", "")),
        "prompt": str(raw.get("prompt") or ""),
        "question": str(raw.get("question") or ""),
        "db_id": str(raw.get("db_id") or ""),
        "gold_sql": str(raw.get("SQL") or ""),
        "difficulty": str(raw.get("difficulty") or "unknown"),
        "schema": str(raw.get("schema") or ""),
        "evidence": str(raw.get("evidence") or ""),
        "task_type": str(raw.get("difficulty") or "unknown"),
    }


class BIRDDataloader(SplitDataLoader):
    """Data loader for BIRD Text-to-SQL benchmark.

    Supports two modes:
    - split_mode="split_dir": read from pre-split train/val/test directories
    - split_mode="ratio": auto-split a single prompt.jsonl by ratio

    Each split directory should contain a prompt.jsonl file.
    """

    def load_split_items(self, split_path: str) -> list[dict]:
        """Load items from one split directory.

        Looks for prompt.jsonl in the directory. Falls back to any .jsonl file.
        """
        path = Path(split_path)

        # Try prompt.jsonl first
        prompt_file = path / "prompt.jsonl"
        if prompt_file.exists():
            return self._load_jsonl(prompt_file)

        # Fall back to any .jsonl file
        jsonl_files = sorted(path.glob("*.jsonl"))
        if jsonl_files:
            return self._load_jsonl(jsonl_files[0])

        # Fall back to .json files
        json_files = sorted(path.glob("*.json"))
        if json_files:
            return self._load_json(json_files[0])

        raise FileNotFoundError(
            f"No prompt.jsonl, .jsonl, or .json file found in {split_path}"
        )

    def load_raw_items(self, data_path: str) -> list[dict]:
        """Load raw items from a prompt.jsonl file for ratio splitting."""
        path = Path(data_path)
        if path.is_dir():
            # Look for prompt.jsonl in directory
            prompt_file = path / "prompt.jsonl"
            if prompt_file.exists():
                return self._load_jsonl(prompt_file)
            jsonl_files = sorted(path.glob("*.jsonl"))
            if jsonl_files:
                return self._load_jsonl(jsonl_files[0])
            raise FileNotFoundError(
                f"No prompt.jsonl found in {data_path}"
            )
        return self._load_jsonl(path)

    def _load_jsonl(self, path: Path) -> list[dict]:
        """Load and normalize items from a JSONL file."""
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(_normalize_item(json.loads(line)))
        return items

    def _load_json(self, path: Path) -> list[dict]:
        """Load and normalize items from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [_normalize_item(row) for row in data]
        raise ValueError(f"Expected JSON array in {path}")
