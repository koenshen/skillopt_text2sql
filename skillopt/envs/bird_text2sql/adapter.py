"""BIRD Text-to-SQL environment adapter for SkillOpt.

Calls BIRD's run_prompt.py and evaluation_v2.py via subprocess,
bridging the two projects without侵入 BIRD's codebase.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from skillopt.envs.base import EnvAdapter
from skillopt.envs.bird_text2sql.dataloader import BIRDDataloader


class BIRDText2SQLAdapter(EnvAdapter):
    """Adapter that bridges SkillOpt training loop to BIRD Text-to-SQL evaluation.

    SkillOpt calls:
      - build_train_env / build_eval_env  →  get batch of questions
      - rollout                            →  call BIRD inference + evaluation
      - reflect                            →  inherited from EnvAdapter
    """

    def __init__(
        self,
        train_prompt_path: str = "",
        dev_prompt_path: str = "",
        db_root_path: str = "",
        bird_project_root: str = "",
        # Model config (passed to run_prompt.py)
        target_base_url: str = "http://10.142.85.18:31181/v1",
        target_api_key: str = "empty",
        target_engine: str = "Qwen3.6-27B",
        target_temperature: float = 1.0,
        target_max_tokens: int = 32768,
        target_timeout: float = 1200,
        target_max_retries: int = 9,
        target_num_processes: int = 100,
        target_max_syntax_attempts: int = 20,
        target_sql_dialect: str = "SQLite",
        # SkillOpt env params
        workers: int = 4,
        analyst_workers: int = 4,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        max_completion_tokens: int = 4096,
        **kwargs,
    ) -> None:
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.max_completion_tokens = int(max_completion_tokens)

        # BIRD-specific config
        self.train_prompt_path = train_prompt_path
        self.dev_prompt_path = dev_prompt_path
        self.db_root_path = os.path.abspath(db_root_path)
        self.bird_project_root = os.path.abspath(bird_project_root)

        # Model config
        self.target_base_url = target_base_url
        self.target_api_key = target_api_key
        self.target_engine = target_engine
        self.target_temperature = target_temperature
        self.target_max_tokens = target_max_tokens
        self.target_timeout = target_timeout
        self.target_max_retries = target_max_retries
        self.target_num_processes = target_num_processes
        self.target_max_syntax_attempts = target_max_syntax_attempts
        self.target_sql_dialect = target_sql_dialect

        # Dataloader
        self.dataloader = BIRDDataloader(
            split_mode="split_dir",
            seed=seed,
            limit=limit,
        )

        # Cache for gold SQL by question_id (loaded per split)
        self._gold_sql_cache: dict[str, str] = {}

    # ── Lifecycle hooks ────────────────────────────────────────────────

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)
        self._load_gold_sql_cache()

    def get_dataloader(self):
        return self.dataloader

    def _load_gold_sql_cache(self) -> None:
        """Load gold SQL from all prompt.jsonl files for evaluation mapping."""
        for path_str in [self.train_prompt_path, self.dev_prompt_path]:
            if not path_str or not os.path.exists(path_str):
                continue
            with open(path_str, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    qid = str(item.get("question_id", ""))
                    gold_sql = item.get("SQL", "")
                    if qid and gold_sql:
                        self._gold_sql_cache[qid] = gold_sql

    # ── Batch → env manager ────────────────────────────────────────────

    def build_env_from_batch(self, batch, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(
            batch_size=batch_size, seed=seed, **kwargs
        )
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(
            env_num=env_num, split=split, seed=seed, **kwargs
        )
        return self.build_env_from_batch(batch, **kwargs)

    # ── Rollout: core logic ────────────────────────────────────────────

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        """Run a batch of Text-to-SQL episodes under the current skill.

        Steps:
          1. Write temp files (skill, prompt.jsonl, diff.json)
          2. Call run_prompt.py → predictions.json
          3. Call evaluation_v2.py → exec_result.jsonl
          4. Parse results and build return dicts
          5. Write trajectory files for reflection
        """
        items = list(env_manager)
        if not items:
            return []

        # Create unique temp directory
        run_id = uuid.uuid4().hex[:12]
        temp_dir = os.path.join(tempfile.gettempdir(), f"bird_text2sql_{run_id}")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            return self._rollout_inner(items, skill_content, out_dir, temp_dir)
        finally:
            # Cleanup temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _rollout_inner(
        self,
        items: list[dict],
        skill_content: str,
        out_dir: str,
        temp_dir: str,
    ) -> list[dict]:
        """Inner rollout logic with guaranteed temp_dir cleanup."""

        # Step 1: Write skill file
        skill_path = os.path.join(temp_dir, "skill.md")
        with open(skill_path, "w", encoding="utf-8") as f:
            f.write(skill_content)

        # Step 2: Write prompt.jsonl (extract fields needed by run_prompt.py)
        prompt_path = os.path.join(temp_dir, "prompt.jsonl")
        with open(prompt_path, "w", encoding="utf-8") as f:
            for item in items:
                row = {
                    "question_id": int(item["id"]) if item["id"].isdigit() else item["id"],
                    "db_id": item.get("db_id", ""),
                    "question": item.get("question", ""),
                    "evidence": item.get("evidence", ""),
                    "full_question": item.get("full_question", ""),
                    "SQL": item.get("gold_sql", ""),
                    "schema": item.get("schema", ""),
                    "prompt": item.get("prompt", ""),
                    "difficulty": item.get("difficulty", "unknown"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Step 3: Write diff.json
        diff_path = os.path.join(temp_dir, "diff.json")
        with open(diff_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"question_id": int(item["id"]) if item["id"].isdigit() else item["id"],
                     "difficulty": item.get("difficulty", "unknown")}
                    for item in items
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Step 4: Call run_prompt.py
        predictions_path = os.path.join(temp_dir, "predictions.json")
        run_prompt_cmd = [
            "python", "-u",
            os.path.join(self.bird_project_root, "src", "run_prompt.py"),
            "--prompt_jsonl", prompt_path,
            "--system_prompt_file", skill_path,
            "--base_url", self.target_base_url,
            "--api_key", self.target_api_key,
            "--engine", self.target_engine,
            "--output_file", predictions_path,
            "--mode", "train",
            "--sql_dialect", self.target_sql_dialect,
            "--temperature", str(self.target_temperature),
            "--max_tokens", str(self.target_max_tokens),
            "--timeout", str(self.target_timeout),
            "--max_retries", str(self.target_max_retries),
            "--num_processes", str(self.target_num_processes),
            "--max_syntax_attempts", str(self.target_max_syntax_attempts),
        ]

        print(f"  [bird_text2sql] rollout: calling run_prompt.py for {len(items)} items")
        result = subprocess.run(
            run_prompt_cmd,
            capture_output=True,
            text=True,
            timeout=int(self.target_timeout) + 60,
            cwd=self.bird_project_root,
        )
        if result.returncode != 0:
            print(f"  [bird_text2sql] run_prompt.py failed:\n{result.stderr[-2000:]}")
            return []

        # Step 5: Read predictions
        if not os.path.exists(predictions_path):
            print(f"  [bird_text2sql] predictions file not found: {predictions_path}")
            return []
        with open(predictions_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        # Step 6: Call evaluation_v2.py
        exec_result_path = os.path.join(temp_dir, "exec_result.jsonl")

        # Find gold SQL file - extract from prompt.jsonl or use provided path
        gold_sql_path = self._create_gold_sql_file(items, temp_dir)

        eval_cmd = [
            "python", "-u",
            os.path.join(self.bird_project_root, "src", "evaluation_v2.py"),
            "--predicted_sql_path", predictions_path,
            "--ground_truth_path", gold_sql_path,
            "--db_root_path", self.db_root_path,
            "--diff_json_path", diff_path,
            "--num_cpus", "1",
            "--meta_time_out", "30.0",
            "--sql_dialect", self.target_sql_dialect,
        ]

        print(f"  [bird_text2sql] rollout: calling evaluation_v2.py")
        result = subprocess.run(
            eval_cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=temp_dir,
        )
        if result.returncode != 0:
            print(f"  [bird_text2sql] evaluation_v2.py failed:\n{result.stderr[-2000:]}")
            return []

        # evaluation_v2.py writes exec_result.jsonl to cwd (temp_dir)
        # But it also writes to the project root. Let's check both locations.
        actual_exec_path = exec_result_path
        if not os.path.exists(actual_exec_path):
            # Check if it was written to bird_project_root
            project_exec = os.path.join(self.bird_project_root, "exec_result.jsonl")
            if os.path.exists(project_exec):
                actual_exec_path = project_exec

        if not os.path.exists(actual_exec_path):
            print(f"  [bird_text2sql] exec_result.jsonl not found")
            return []

        # Step 7: Parse exec_result.jsonl
        exec_results = []
        with open(actual_exec_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    exec_results.append(json.loads(line))

        # Build mapping: sql_idx → exec_result
        # sql_idx corresponds to the order in prompt.jsonl
        exec_by_idx = {r["sql_idx"]: r for r in exec_results}

        # Step 8: Build return values
        results = []
        trajectory_dir = os.path.join(out_dir, "predictions")
        os.makedirs(trajectory_dir, exist_ok=True)

        for idx, item in enumerate(items):
            qid = item["id"]
            pred_str = predictions.get(qid, "")
            exec_r = exec_by_idx.get(idx, {})

            # Parse predicted SQL from BIRD format: "SQL\t----- bird -----\tdb_id"
            predicted_sql = pred_str
            if "\t----- bird -----\t" in pred_str:
                predicted_sql = pred_str.split("\t----- bird -----\t")[0]

            # Execution result
            res = exec_r.get("res", 0)
            predicted_res = exec_r.get("predicted_res")
            ground_truth_res = exec_r.get("ground_truth_res")
            gold_sql_from_eval = exec_r.get("ground_truth", item.get("gold_sql", ""))

            # Build rollout result
            rollout_result = {
                "id": qid,
                "hard": int(res),
                "soft": float(res),
                "predicted_answer": predicted_sql,
                "question": item.get("question", ""),
                "task_type": item.get("difficulty", "unknown"),
                "extras": {
                    "db_id": item.get("db_id", ""),
                    "predicted_sql": predicted_sql,
                    "gold_sql": gold_sql_from_eval,
                    "predicted_result": predicted_res,
                    "gold_result": ground_truth_res,
                    "difficulty": item.get("difficulty", "unknown"),
                },
            }
            results.append(rollout_result)

            # Step 9: Write trajectory file for reflection
            qid_dir = os.path.join(trajectory_dir, qid)
            os.makedirs(qid_dir, exist_ok=True)
            trajectory = [
                {"role": "system", "content": skill_content},
                {"role": "user", "content": item.get("prompt", "")},
                {"role": "assistant", "content": pred_str},
                {
                    "role": "system",
                    "content": (
                        f"Execution result:\n"
                        f"- Predicted SQL: {predicted_sql}\n"
                        f"- Gold SQL: {gold_sql_from_eval}\n"
                        f"- Match: {'Yes' if res else 'No'}\n"
                        f"- Predicted result: {json.dumps(predicted_res, default=str)[:500]}\n"
                        f"- Gold result: {json.dumps(ground_truth_res, default=str)[:500]}"
                    ),
                },
            ]
            with open(os.path.join(qid_dir, "conversation.json"), "w", encoding="utf-8") as f:
                json.dump(trajectory, f, ensure_ascii=False, indent=2)

        # Summary
        n_correct = sum(1 for r in results if r["hard"])
        print(
            f"  [bird_text2sql] rollout: {n_correct}/{len(results)} correct "
            f"({n_correct/len(results)*100:.1f}%)"
        )

        return results

    def _create_gold_sql_file(self, items: list[dict], temp_dir: str) -> str:
        """Create a gold.sql file for evaluation_v2.py.

        Format: SQL<TAB>db_id (one line per question, in order).
        """
        gold_path = os.path.join(temp_dir, "gold.sql")
        with open(gold_path, "w", encoding="utf-8") as f:
            for item in items:
                gold_sql = item.get("gold_sql", "")
                db_id = item.get("db_id", "")
                f.write(f"{gold_sql}\t{db_id}\n")
        return gold_path

    # ── Stratification hint ────────────────────────────────────────────

    def get_task_types(self) -> list[str]:
        """Distinct difficulty levels used for stratified sampling."""
        seen = []
        all_items = (
            self.dataloader.train_items
            + self.dataloader.val_items
            + self.dataloader.test_items
        )
        for item in all_items:
            tt = str(item.get("difficulty") or item.get("task_type") or "unknown")
            if tt not in seen:
                seen.append(tt)
        return seen or ["unknown"]
