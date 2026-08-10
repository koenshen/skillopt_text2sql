"""Run the existing Text-to-SQL Deep Agent and score it with BIRD EX."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from skillopt.envs.text2sql.evaluator import evaluate_one
from skillopt.envs.text2sql.agent_bridge import (
    HOST_PROJECT_ROOT,
    build_bird_question,
    create_model_from_config,
    create_skillopt_agent,
    invoke_and_extract,
    load_model_config,
)


PROJECT_ROOT = HOST_PROJECT_ROOT


class _RolloutProgress:
    """Thread-safe, line-oriented progress reporting for long Agent rollouts."""

    def __init__(
        self,
        total: int,
        groups: int,
        workers: int,
        out_root: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.total = total
        self.groups = groups
        self.workers = workers
        self.started = 0
        self.completed = 0
        self.correct = 0
        self.batch_started_at = time.monotonic()
        self.lock = Lock()
        self.context = dict(context or {})
        phase = str(self.context.get("phase") or Path(out_root).name or out_root)
        self.context["phase"] = phase
        self._print(
            f"[Text2SQL rollout] {self._context_text()} items={total} "
            f"database_groups={groups} workers={workers}"
        )

    def _context_text(self) -> str:
        fields = [f"phase={self.context['phase']}"]
        epoch = self.context.get("epoch")
        num_epochs = self.context.get("num_epochs")
        if epoch is not None and num_epochs is not None:
            fields.append(
                "epoch=pretrain"
                if int(epoch) == 0
                else f"epoch={epoch}/{num_epochs}"
            )
        global_step = self.context.get("global_step")
        total_steps = self.context.get("total_steps")
        if global_step is not None and total_steps is not None:
            fields.append(f"step={global_step}/{total_steps}")
        step_in_epoch = self.context.get("step_in_epoch")
        steps_per_epoch = self.context.get("steps_per_epoch")
        if step_in_epoch is not None and steps_per_epoch is not None:
            fields.append(f"epoch_step={step_in_epoch}/{steps_per_epoch}")
        batch_in_epoch = self.context.get("batch_in_epoch")
        batches_per_epoch = self.context.get("batches_per_epoch")
        if batch_in_epoch is not None and batches_per_epoch is not None:
            fields.append(f"batch={batch_in_epoch}/{batches_per_epoch}")
        accumulation_index = self.context.get("accumulation_index")
        accumulation = self.context.get("accumulation")
        if accumulation_index is not None and accumulation is not None:
            fields.append(f"accum={accumulation_index}/{accumulation}")
        remaining_steps = self.context.get("remaining_steps")
        if remaining_steps is not None:
            fields.append(f"steps_after_current={remaining_steps}")
        return " ".join(fields)

    def _item_prefix(self) -> str:
        return f"[{self._context_text()}]"

    @staticmethod
    def _print(message: str) -> None:
        print(f"  {message}", flush=True)

    def group_start(self, split: str, db_id: str, count: int) -> float:
        started_at = time.monotonic()
        with self.lock:
            self._print(
                f"{self._item_prefix()} [DB setup] START "
                f"split={split} db={db_id} items={count}"
            )
        return started_at

    def group_ready(self, split: str, db_id: str, started_at: float) -> None:
        elapsed = time.monotonic() - started_at
        with self.lock:
            self._print(
                f"{self._item_prefix()} [DB setup] READY "
                f"split={split} db={db_id} elapsed={elapsed:.1f}s"
            )

    def item_start(self, item: dict[str, Any]) -> tuple[int, float]:
        started_at = time.monotonic()
        with self.lock:
            self.started += 1
            ordinal = self.started
            self._print(
                f"{self._item_prefix()} [Question {ordinal}/{self.total}] START "
                f"id={item['id']} split={item['source_split']} "
                f"db={item['db_id']} difficulty={item.get('task_type', 'text2sql')}"
            )
        return ordinal, started_at

    def item_done(
        self,
        item: dict[str, Any],
        ordinal: int,
        started_at: float,
        *,
        status: str,
        hard: int,
        turns: int,
    ) -> None:
        elapsed = time.monotonic() - started_at
        with self.lock:
            self.completed += 1
            self.correct += int(hard)
            accuracy = self.correct / self.completed if self.completed else 0.0
            remaining = self.total - self.completed
            self._print(
                f"{self._item_prefix()} [Question {ordinal}/{self.total}] DONE "
                f"id={item['id']} status={status} hard={hard} turns={turns} "
                f"elapsed={elapsed:.1f}s progress={self.completed}/{self.total} "
                f"remaining_questions={remaining} "
                f"running_accuracy={accuracy:.3f}"
            )

    def batch_done(self) -> None:
        elapsed = time.monotonic() - self.batch_started_at
        accuracy = self.correct / self.completed if self.completed else 0.0
        with self.lock:
            self._print(
                f"[Text2SQL rollout] {self._context_text()} COMPLETE "
                f"items={self.completed}/{self.total} "
                f"correct={self.correct} accuracy={accuracy:.3f} elapsed={elapsed:.1f}s"
            )


def _clip_repr(value: Any, limit: int = 4000) -> str:
    rendered = repr(value)
    if len(rendered) <= limit:
        return rendered
    return rendered[:limit] + f"... [truncated {len(rendered) - limit} chars]"


def _runtime_paths(source_split: str, db_id: str) -> tuple[str, Path, Path]:
    if source_split == "train":
        topic = "bird_train"
        database_root = PROJECT_ROOT / "bird_train_datas/train_databases"
        schema_root = PROJECT_ROOT / "topics/bird_train/schemas"
    elif source_split == "dev":
        # Gate uses Full Dev after Mini-dev IDs were excluded during data
        # materialization. Mini-dev's topic, schema, and databases are never
        # loaded by a SkillOpt training run.
        topic = "bird_train"
        database_root = PROJECT_ROOT / "data_dev/dev_databases"
        schema_root = (
            Path(__file__).resolve().parents[3]
            / "data/bird_text2sql/schemas/dev"
        )
    else:
        raise ValueError(f"unsupported Text-to-SQL split: {source_split}")
    db_path = database_root / db_id / f"{db_id}.sqlite"
    return topic, database_root, schema_root


def _conversation_from_steps(
    question: str,
    steps: list[dict[str, Any]],
    final_answer: str,
    evaluation: dict[str, Any],
) -> list[dict[str, Any]]:
    conversation: list[dict[str, Any]] = [{"role": "user", "content": question}]
    for step in steps:
        thinking = str(step.get("ai_thinking") or "")
        if thinking:
            conversation.append({"role": "assistant", "content": thinking})
        tool_calls = step.get("tool_calls", [])
        tool_results = step.get("tool_results", [])
        for index, tool_call in enumerate(tool_calls):
            result = tool_results[index] if index < len(tool_results) else {}
            command = f"{tool_call.get('name', '')}({json.dumps(tool_call.get('args', {}), ensure_ascii=False)})"
            conversation.append(
                {
                    "type": "tool_call",
                    "cmd": command,
                    "obs": str(result.get("content") or ""),
                }
            )
    if final_answer:
        conversation.append({"role": "assistant", "content": final_answer})
    verification = (
        "[BIRD OFFICIAL EXECUTION RESULT]\n"
        f"Status: {evaluation['status']}\n"
        f"Execution correct: {evaluation['res']}\n"
        f"Predicted SQL: {evaluation['predicted_sql']}\n"
        f"Ground-truth SQL: {evaluation['ground_truth']}\n"
        f"Predicted rows: {_clip_repr(evaluation.get('predicted_res'))}\n"
        f"Ground-truth rows: {_clip_repr(evaluation.get('ground_truth_res'))}\n"
        f"Error: {evaluation.get('error', '')}"
    )
    conversation.append({"role": "system", "content": verification})
    return conversation


def _save_artifacts(
    out_root: str,
    item_id: str,
    skill_content: str,
    question: str,
    conversation: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> None:
    prediction_dir = Path(out_root) / "predictions" / item_id
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (prediction_dir / "target_system_prompt.txt").write_text(
        skill_content, encoding="utf-8"
    )
    (prediction_dir / "target_user_prompt.txt").write_text(question, encoding="utf-8")
    with (prediction_dir / "conversation.json").open("w", encoding="utf-8") as handle:
        json.dump(conversation, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    with (prediction_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def _failure_result(item: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "id": str(item["id"]),
        "hard": 0,
        "soft": 0.0,
        "question": item.get("question", ""),
        "task_description": item.get("question", ""),
        "task_type": item.get("task_type", "text2sql"),
        "predicted_answer": "",
        "predicted_sql": "",
        "ground_truth_sql": item.get("ground_truth_sql", ""),
        "reference_text": f"Ground-truth SQL: {item.get('ground_truth_sql', '')}",
        "fail_reason": reason,
        "agent_ok": False,
        "n_turns": 0,
    }


def _run_database_group(
    items: list[dict[str, Any]],
    out_root: str,
    skill_content: str,
    execution_timeout: float,
    agent_model_config_name: str,
    progress: _RolloutProgress,
) -> list[dict[str, Any]]:
    first = items[0]
    db_id = str(first["db_id"])
    source_split = str(first["source_split"])
    topic, database_root, schema_root = _runtime_paths(source_split, db_id)
    group_started_at = progress.group_start(source_split, db_id, len(items))
    try:
        model_config = load_model_config()
        configured_name = (
            agent_model_config_name
            or model_config.get("default_model", "")
        )
        if not configured_name:
            raise ValueError(
                "Agent model is not configured in the SkillOpt environment or "
                "as default_model in model_config.yaml"
            )
        agent_model = create_model_from_config(configured_name, model_config)
        agent = create_skillopt_agent(
            topic=topic,
            model=agent_model,
            db_id=db_id,
            schema_root=schema_root,
            skill_content=skill_content,
            database_root=database_root,
        )
        progress.group_ready(source_split, db_id, group_started_at)
    except (Exception, SystemExit) as exc:
        reason = f"agent_creation_error: {type(exc).__name__}: {exc}"
        failures = []
        for item in items:
            ordinal, started_at = progress.item_start(item)
            failures.append(_failure_result(item, reason))
            progress.item_done(
                item,
                ordinal,
                started_at,
                status=reason,
                hard=0,
                turns=0,
            )
        return failures

    results: list[dict[str, Any]] = []
    db_path = database_root / db_id / f"{db_id}.sqlite"
    for item in items:
        ordinal, item_started_at = progress.item_start(item)
        item_id = str(item["id"])
        question = build_bird_question(item)
        try:
            final_answer, steps, predicted_sql = invoke_and_extract(agent, question)
            evaluation = evaluate_one(
                predicted_sql=predicted_sql,
                ground_truth_sql=item["ground_truth_sql"],
                db_path=db_path,
                timeout=execution_timeout,
            )
            conversation = _conversation_from_steps(
                question, steps, final_answer, evaluation
            )
            _save_artifacts(
                out_root,
                item_id,
                skill_content,
                question,
                conversation,
                evaluation,
            )
            hard = int(evaluation["res"])
            turns = len(steps)
            results.append(
                {
                    "id": item_id,
                    "hard": hard,
                    "soft": float(hard),
                    "question": item["question"],
                    "task_description": item["question"],
                    "task_type": item.get("task_type", "text2sql"),
                    "predicted_answer": final_answer,
                    "predicted_sql": predicted_sql,
                    "ground_truth_sql": item["ground_truth_sql"],
                    "reference_text": (
                        f"Ground-truth SQL: {item['ground_truth_sql']}\n"
                        f"BIRD execution status: {evaluation['status']}"
                    ),
                    "fail_reason": "" if hard else evaluation["status"],
                    "agent_ok": True,
                    "n_turns": turns,
                    "target_system_prompt": skill_content,
                    "target_user_prompt": question,
                    "db_id": db_id,
                    "execution_status": evaluation["status"],
                }
            )
            progress.item_done(
                item,
                ordinal,
                item_started_at,
                status=str(evaluation["status"]),
                hard=hard,
                turns=turns,
            )
        except Exception as exc:
            reason = f"agent_error: {type(exc).__name__}: {exc}"
            result = _failure_result(
                item, reason
            )
            results.append(result)
            progress.item_done(
                item,
                ordinal,
                item_started_at,
                status=reason,
                hard=0,
                turns=0,
            )
            if isinstance(exc, TypeError) and "unexpected keyword argument" in str(exc):
                raise RuntimeError(
                    "Fatal Text-to-SQL Agent model configuration error; aborting "
                    f"the rollout instead of recording infrastructure failures as reward 0: {exc}"
                ) from exc
    return results


def run_batch(
    items: list[dict[str, Any]],
    out_root: str,
    skill_content: str,
    execution_timeout: float = 30.0,
    workers: int = 1,
    agent_model_config_name: str = "",
    progress_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one Agent per database, sequentially within each database group."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[(str(item["source_split"]), str(item["db_id"]))].append(item)

    group_list = list(groups.values())
    effective_workers = min(max(int(workers), 1), max(len(group_list), 1))
    progress = _RolloutProgress(
        total=len(items),
        groups=len(group_list),
        workers=effective_workers,
        out_root=out_root,
        context=progress_context,
    )
    if workers <= 1 or len(group_list) <= 1:
        nested = [
            _run_database_group(
                group,
                out_root,
                skill_content,
                execution_timeout,
                agent_model_config_name,
                progress,
            )
            for group in group_list
        ]
    else:
        nested = []
        with ThreadPoolExecutor(max_workers=min(workers, len(group_list))) as executor:
            futures = {
                executor.submit(
                    _run_database_group,
                    group,
                    out_root,
                    skill_content,
                    execution_timeout,
                    agent_model_config_name,
                    progress,
                ): group
                for group in group_list
            }
            for future in as_completed(futures):
                nested.append(future.result())

    by_id = {result["id"]: result for group_results in nested for result in group_results}
    progress.batch_done()
    return [by_id[str(item["id"])] for item in items]
