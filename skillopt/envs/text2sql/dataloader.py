"""Train/Dev loader for the preprocessed BIRD Text-to-SQL data."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from skillopt.datasets.base import BatchSpec, SplitDataLoader


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Text2SQLDataLoader(SplitDataLoader):
    """Load public prompts and private SQL by stable ID, never by line join."""

    def setup(self, cfg: dict) -> None:
        split_dir = Path(self.split_dir or cfg.get("split_dir", ""))
        if not split_dir.is_absolute():
            skillopt_root = Path(__file__).resolve().parents[3]
            candidate = skillopt_root / split_dir
            split_dir = candidate if candidate.exists() else Path.cwd() / split_dir
        split_dir = split_dir.resolve()
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"BIRD Text-to-SQL materialization not found: {split_dir}. "
                "Run scripts/prepare_bird_skillopt_data.py from the repository root."
            )

        train = self._join_public_private(
            split_dir / "public/train.jsonl",
            split_dir / "private/train_gold.jsonl",
        )
        dev = self._join_public_private(
            split_dir / "public/dev.jsonl",
            split_dir / "private/dev_gold.jsonl",
        )

        train_order_payload = json.loads(
            (split_dir / "selections/train_order.json").read_text(encoding="utf-8")
        )
        train_ids = [str(item["id"]) for item in train_order_payload["ordered_items"]]
        train_by_id = {str(item["id"]): item for item in train}
        if set(train_ids) != set(train_by_id) or len(train_ids) != len(train):
            raise ValueError("Train ordering does not exactly cover the Train records")
        train = [train_by_id[item_id] for item_id in train_ids]

        gate_payload = json.loads(
            (split_dir / "selections/dev_gate_order.json").read_text(encoding="utf-8")
        )
        gate_ids = [str(item["id"]) for item in gate_payload["ordered_items"]]
        dev_by_id = {str(item["id"]): item for item in dev}
        if set(gate_ids) != set(dev_by_id) or len(gate_ids) != len(dev):
            raise ValueError("Dev Gate ordering does not exactly cover the Dev records")
        ordered_dev = [dev_by_id[item_id] for item_id in gate_ids]

        configured_train_size = int(cfg.get("train_size", 0) or 0)
        if configured_train_size:
            if configured_train_size > len(train):
                raise ValueError(
                    f"train_size={configured_train_size} exceeds available Train records={len(train)}"
                )
            train = train[:configured_train_size]

        for item in train:
            item["task_type"] = "text2sql"
        for item in ordered_dev:
            item["task_type"] = str(item.get("difficulty") or "text2sql")

        self.split_dir = str(split_dir)
        self._splits = {"train": train, "val": ordered_dev, "test": []}
        print(
            f"  [{type(self).__name__}] train={len(train)} val={len(ordered_dev)} "
            f"test=0 (Test intentionally disabled)"
        )

    def plan_train_epoch(
        self,
        *,
        epoch: int,
        steps_per_epoch: int,
        accumulation: int,
        batch_size: int,
        seed: int,
        **kwargs,
    ) -> list[BatchSpec]:
        """Cover the selected Train pool exactly once without refill samples."""
        items = list(self.train_items)
        random.Random(seed + epoch * 1000).shuffle(items)
        total_batches = steps_per_epoch * accumulation
        if total_batches <= 0:
            return []

        base_size, extra = divmod(len(items), total_batches)
        if base_size + (1 if extra else 0) > batch_size:
            raise ValueError(
                "Text2SQL epoch plan cannot fit the Train pool into the configured batches"
            )

        batches: list[BatchSpec] = []
        cursor = 0
        for batch_idx in range(total_batches):
            current_size = base_size + (1 if batch_idx < extra else 0)
            batch_items = items[cursor: cursor + current_size]
            cursor += current_size
            batches.append(
                BatchSpec(
                    phase="train",
                    split="train",
                    seed=seed + epoch * 1000 + batch_idx + 1,
                    batch_size=current_size,
                    payload=batch_items,
                    metadata={"epoch": epoch, "batch_index": batch_idx},
                )
            )
        if cursor != len(items):
            raise AssertionError("Text2SQL epoch plan did not consume the exact Train pool")
        return batches

    @staticmethod
    def _join_public_private(public_path: Path, private_path: Path) -> list[dict[str, Any]]:
        public = load_jsonl(public_path)
        private = load_jsonl(private_path)
        private_by_id = {str(item["id"]): item for item in private}
        if len(private_by_id) != len(private):
            raise ValueError(f"duplicate private IDs in {private_path}")
        public_ids = [str(item["id"]) for item in public]
        if len(set(public_ids)) != len(public_ids):
            raise ValueError(f"duplicate public IDs in {public_path}")
        if set(public_ids) != set(private_by_id):
            raise ValueError(f"public/private ID mismatch: {public_path} vs {private_path}")

        joined = []
        for item in public:
            item_id = str(item["id"])
            private_item = private_by_id[item_id]
            if item["db_id"] != private_item["db_id"]:
                raise ValueError(f"database mismatch for {item_id}")
            joined.append({**item, "ground_truth_sql": private_item["ground_truth_sql"]})
        return joined
