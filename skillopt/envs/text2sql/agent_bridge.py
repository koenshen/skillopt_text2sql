"""Thin compatibility bridge to the host Text-to-SQL Agent framework.

This module deliberately contains no SkillOpt training, selection, reward, or
data-splitting logic. It centralizes the small, version-sensitive boundary to
the continuously updated host ``agent.py`` and its proven ``eval.py`` SQL
extraction helpers.
"""
from __future__ import annotations

import inspect
import os
import sys
import uuid
from pathlib import Path
from typing import Any


_SKILLOPT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOST_PROJECT_ROOT = Path(
    os.environ.get(
        "TEXT2SQL_AGENT_ROOT",
        _SKILLOPT_PROJECT_ROOT.parent / "text-to-sql-agent",
    )
).resolve()
if not (HOST_PROJECT_ROOT / "agent.py").is_file():
    raise RuntimeError(
        "Text2SQL host project was not found at "
        f"{HOST_PROJECT_ROOT}. Set TEXT2SQL_AGENT_ROOT to the directory "
        "containing agent.py."
    )
if str(HOST_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(HOST_PROJECT_ROOT))

import agent as host_agent  # noqa: E402
import eval as host_eval  # noqa: E402
from tools.custom_schema import current_user_question  # noqa: E402


_REQUIRED_AGENT_FUNCTIONS = (
    "load_model_config",
    "create_model_from_config",
    "create_sql_deep_agent",
)
_REQUIRED_AGENT_PARAMETERS = (
    "custom_schema_dir",
    "learned_skill",
    "database_root_override",
)
_REQUIRED_EVAL_FUNCTIONS = (
    "build_bird_question",
    "extract_final_answer",
    "extract_intermediate_steps",
    "extract_sql_from_bird_answer",
    "extract_sql_from_steps",
)


def validate_skillopt_agent_compatibility() -> None:
    """Fail fast when the host Agent no longer satisfies SkillOpt's contract.

    This check is read-only. It creates no Agent, sends no model request,
    executes no SQL, and modifies no files.
    """
    missing_functions = [
        name
        for name in _REQUIRED_AGENT_FUNCTIONS
        if not callable(getattr(host_agent, name, None))
    ]
    if missing_functions:
        raise RuntimeError(
            "Text2SQL SkillOpt compatibility check failed. The current agent.py "
            f"is missing callable functions: {', '.join(missing_functions)}. "
            "No model request, SQL execution, or training step has started."
        )

    signature = inspect.signature(host_agent.create_sql_deep_agent)
    missing_parameters = [
        name for name in _REQUIRED_AGENT_PARAMETERS if name not in signature.parameters
    ]
    if missing_parameters:
        raise RuntimeError(
            "Text2SQL SkillOpt compatibility check failed. "
            "create_sql_deep_agent() in the current agent.py is missing SkillOpt "
            f"integration parameters: {', '.join(missing_parameters)}. "
            "No model request, SQL execution, or training step has started."
        )

    missing_extractors = [
        name
        for name in _REQUIRED_EVAL_FUNCTIONS
        if not callable(getattr(host_eval, name, None))
    ]
    if missing_extractors:
        raise RuntimeError(
            "Text2SQL SkillOpt compatibility check failed. The current eval.py "
            f"is missing callable helpers: {', '.join(missing_extractors)}. "
            "No model request, SQL execution, or training step has started."
        )


def load_model_config() -> dict[str, Any]:
    """Load the host Agent's model registry without maintaining a copy."""
    return host_agent.load_model_config()


def create_model_from_config(model_name: str, config: dict[str, Any]) -> Any:
    """Create a model through the current host Agent implementation."""
    return host_agent.create_model_from_config(model_name, config)


def create_skillopt_agent(
    *,
    topic: str,
    model: Any,
    db_id: str,
    schema_root: Path,
    database_root: Path,
    skill_content: str,
) -> Any:
    """Create the current host Agent with only the dedicated SkillOpt hooks."""
    agent_instance, _ = host_agent.create_sql_deep_agent(
        topic=topic,
        model=model,
        lang="en",
        db_id=db_id,
        custom_schema_dir=str(schema_root),
        learned_skill=skill_content,
        database_root_override=str(database_root),
    )
    return agent_instance


def build_bird_question(item: dict[str, Any]) -> str:
    """Build the prompt with the host evaluator's existing BIRD logic."""
    return host_eval.build_bird_question(item)


def invoke_and_extract(
    agent_instance: Any,
    question: str,
) -> tuple[str, list[dict[str, Any]], str]:
    """Invoke the current Agent and reuse the host's existing extraction logic."""
    token = current_user_question.set(question)
    try:
        response = agent_instance.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
    finally:
        current_user_question.reset(token)

    messages = response.get("messages", [])
    final_answer = host_eval.extract_final_answer(messages)
    steps = host_eval.extract_intermediate_steps(messages)
    predicted_sql = host_eval.extract_sql_from_bird_answer(final_answer)
    if not predicted_sql:
        predicted_sql = host_eval.extract_sql_from_steps(steps)
    return final_answer, steps, predicted_sql
