"""SkillOpt-Sleep — consolidate several skill groups in one night.

A night normally consolidates one managed skill. When mined tasks have been
grouped per skill, each group is an independent consolidation: its own tasks, its
own document, its own gate decision. This module drives those runs so that a
group with no tasks, an unusable name, or a failing backend is isolated and
reported instead of aborting the groups around it.

Adoption is unchanged and still explicit: nothing here writes a skill file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from skillopt_sleep.backend import Backend
from skillopt_sleep.consolidate import ConsolidationResult, consolidate
from skillopt_sleep.types import TaskRecord

CONSOLIDATED = "consolidated"
SKIPPED = "skipped"
FAILED = "failed"


@dataclass
class SkillGroup:
    """One skill's slice of the night: its name, its document, its tasks."""

    skill_name: str
    skill: str = ""
    tasks: List[TaskRecord] = field(default_factory=list)


@dataclass
class GroupConsolidation:
    """Per-group outcome. ``result`` is set only when the group actually ran."""

    skill_name: str
    status: str
    result: Optional[ConsolidationResult] = None
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return bool(self.result and self.result.accepted)


def consolidate_groups(
    backend: Backend,
    groups: Sequence[SkillGroup],
    memory: str = "",
    *,
    consolidate_fn: Callable[..., ConsolidationResult] = consolidate,
    **consolidate_kwargs: object,
) -> Dict[str, GroupConsolidation]:
    """Consolidate each group independently, in order, isolating failures.

    Returns one entry per distinct skill name, keyed by that name and ordered
    first-seen — at most one entry per input group, not exactly one. The result
    is keyed by name, so a repeated name structurally cannot carry a second
    outcome: the first group under a name wins and later ones are dropped
    without an entry of their own. Every blank name likewise collapses into the
    single ``""`` entry. Callers needing a decision for every input must pass
    distinct, non-blank names.

    A group is ``skipped`` when it is unusable (blank name, or no tasks) and
    ``failed`` when its own consolidation raised; both leave every other
    group's decision untouched.

    ``memory`` is the shared agent memory and is passed through read-only: group
    runs evolve skills only, so no group can rewrite another group's memory.
    """
    # This wrapper's contract is stricter than consolidate(): shared memory is
    # always read-only.  Override a caller-supplied value instead of passing a
    # duplicate keyword (which would otherwise turn the group into a failure).
    consolidate_kwargs["evolve_memory"] = False
    out: Dict[str, GroupConsolidation] = {}
    for group in groups:
        name = (group.skill_name or "").strip()
        if not name:
            out.setdefault("", GroupConsolidation("", SKIPPED, reason="group has no skill name"))
            continue
        if name in out:
            continue  # first group wins; a repeated name is not a second night
        if not group.tasks:
            out[name] = GroupConsolidation(name, SKIPPED, reason="no mined tasks for this skill")
            continue
        try:
            result = consolidate_fn(
                backend, list(group.tasks), group.skill, memory,
                **consolidate_kwargs,
            )
        except Exception as exc:  # one group's failure must not abort the night
            out[name] = GroupConsolidation(
                name, FAILED, reason=f"{type(exc).__name__}: {exc}"[:300]
            )
            continue
        out[name] = GroupConsolidation(name, CONSOLIDATED, result=result)
    return out


def accepted_group_skills(outcomes: Dict[str, GroupConsolidation]) -> Dict[str, str]:
    """Skill name -> new document, for groups whose gate accepted an update."""
    return {
        name: outcome.result.new_skill
        for name, outcome in outcomes.items()
        if outcome.accepted and outcome.result is not None
    }
