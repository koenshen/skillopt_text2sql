"""SkillOpt adapter for the existing Text-to-SQL Deep Agent."""
from __future__ import annotations

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.text2sql.agent_bridge import (
    load_model_config,
    validate_skillopt_agent_compatibility,
)
from skillopt.envs.text2sql.dataloader import Text2SQLDataLoader
from skillopt.envs.text2sql.rollout import run_batch


class Text2SQLAdapter(EnvAdapter):
    def __init__(
        self,
        split_dir: str = "data/bird_text2sql",
        split_mode: str = "split_dir",
        split_seed: int = 42,
        workers: int = 1,
        analyst_workers: int = 1,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        execution_timeout: float = 30.0,
        agent_model_config_name: str = "",
        optimizer_model_config_name: str = "",
    ) -> None:
        self.workers = int(workers)
        self.analyst_workers = int(analyst_workers)
        self.failure_only = bool(failure_only)
        self.minibatch_size = int(minibatch_size)
        self.edit_budget = int(edit_budget)
        self.execution_timeout = float(execution_timeout)
        self.agent_model_config_name = str(agent_model_config_name or "")
        self.optimizer_model_config_name = str(optimizer_model_config_name or "")
        self.dataloader = Text2SQLDataLoader(
            split_dir=split_dir,
            split_mode=split_mode,
            split_seed=split_seed,
            seed=seed,
            limit=limit,
        )

    def setup(self, cfg: dict) -> None:
        """Initialize the SkillOpt Text-to-SQL environment.

        Before loading data or issuing any model request, perform a read-only
        compatibility check against the host Text-to-SQL Agent. The check does
        not create an Agent, execute SQL, or modify any host framework file.
        """
        if getattr(self, "_setup_complete", False):
            return
        # Text2SQL-specific SkillOpt host preflight. Keeping this in the
        # environment adapter leaves the shared SkillOpt train.py untouched.
        validate_skillopt_agent_compatibility()
        self._configure_optimizer_from_agent_model_config(cfg)
        super().setup(cfg)
        self.dataloader.setup(cfg)
        self._setup_complete = True

    def _configure_optimizer_from_agent_model_config(self, cfg: dict) -> None:
        """Map the existing root model_config.yaml entry to SkillOpt's chat backend."""
        model_config = load_model_config()
        agent_config_name = (
            self.agent_model_config_name
            or model_config.get("default_model", "")
        )
        optimizer_config_name = (
            self.optimizer_model_config_name
            or model_config.get("default_model", "")
        )
        models = model_config.get("models", {})
        agent_entry = models.get(agent_config_name)
        optimizer_entry = models.get(optimizer_config_name)
        if not agent_entry:
            raise ValueError(
                f"Agent model_config entry {agent_config_name!r} was not found in model_config.yaml"
            )
        if not optimizer_entry:
            raise ValueError(
                f"optimizer model_config entry {optimizer_config_name!r} was not found in model_config.yaml"
            )
        if not optimizer_entry.get("base_url") or not optimizer_entry.get("model_name"):
            raise ValueError(
                f"optimizer model_config entry {optimizer_config_name!r} must provide "
                "base_url and model_name"
            )

        extra_body = optimizer_entry.get("extra_body") or {}
        chat_template = extra_body.get("chat_template_kwargs") or {}
        enable_thinking = bool(chat_template.get("enable_thinking", False))
        temperature = optimizer_entry.get("temperature")

        cfg.update(
            {
                "backend": "qwen_chat",
                "model_backend": "qwen_chat",
                "optimizer_backend": "qwen_chat",
                "target_backend": "qwen_chat",  # Target chat path is unused by this adapter.
                "optimizer_model": str(optimizer_entry["model_name"]),
                "target_model": str(agent_entry.get("model_name") or agent_config_name),
                "optimizer_qwen_chat_base_url": str(optimizer_entry["base_url"]),
                "optimizer_qwen_chat_api_key": str(optimizer_entry.get("api_key") or ""),
                "optimizer_qwen_chat_temperature": (
                    temperature if temperature is not None else ""
                ),
                "optimizer_qwen_chat_timeout_seconds": int(
                    optimizer_entry.get("timeout") or 1200
                ),
                "optimizer_qwen_chat_max_tokens": int(
                    optimizer_entry.get("max_tokens")
                    or extra_body.get("max_completion_tokens")
                    or 16384
                ),
                "optimizer_qwen_chat_enable_thinking": enable_thinking,
            }
        )
        print(
            f"  [Text2SQLAdapter] Target Agent model: {agent_config_name} "
            f"({agent_entry.get('model_name', agent_config_name)})\n"
            f"  [Text2SQLAdapter] SkillOpt Optimizer model: {optimizer_config_name} "
            f"({optimizer_entry['model_name']})"
        )

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size, seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(env_num, split, seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(self, env_manager, skill_content: str, out_dir: str, **kwargs) -> list[dict]:
        return run_batch(
            items=list(env_manager),
            out_root=out_dir,
            skill_content=skill_content,
            execution_timeout=self.execution_timeout,
            workers=self.workers,
            agent_model_config_name=self.agent_model_config_name,
            progress_context=kwargs.get("progress_context"),
        )

    def get_task_types(self) -> list[str]:
        return ["text2sql", "simple", "moderate", "challenging"]
