"""Explicit outcome labeling without semantic fallbacks between signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from rpent.research.handoff.types import (
    LabelSource,
    OutcomeLabels,
    OutcomeSignal,
    SkillIdentity,
    unavailable_signal,
)


@dataclass(frozen=True)
class LabelContext:
    skill: SkillIdentity
    primitive_result: Mapping[str, Any]
    episode_terminated: bool
    episode_truncated: bool
    llm_finish_status: str | None = None


@runtime_checkable
class OutcomeLabeler(Protocol):
    def label(self, context: LabelContext) -> OutcomeLabels:
        """Produce separate labels; unavailable signals stay unavailable."""


def _boolean_or_unknown(
    value: Any,
    *,
    source: LabelSource,
    definition: str,
    evaluator_id: str | None = None,
) -> OutcomeSignal:
    if isinstance(value, bool):
        return OutcomeSignal(
            value=value,
            source=source,
            definition=definition,
            evaluator_id=evaluator_id,
        )
    return unavailable_signal(definition)


class PrimitiveAndOfficialOutcomeLabeler:
    """Default non-oracle labeler matching current RPent result semantics."""

    def label(self, context: LabelContext) -> OutcomeLabels:
        primitive = _boolean_or_unknown(
            context.primitive_result.get("success"),
            source=LabelSource.PRIMITIVE_HEURISTIC,
            definition=(
                "primitive-specific RPent result; for pi0_pick this is the "
                "descent/lift/gripper heuristic and is not correct-object proof"
            ),
        )
        skill_value = context.primitive_result.get("skill_success")
        skill = _boolean_or_unknown(
            skill_value,
            source=LabelSource.SKILL_EVALUATOR,
            definition="explicit skill-specific evaluator result",
        )
        task = OutcomeSignal(
            value=bool(context.episode_terminated),
            source=LabelSource.OFFICIAL_TERMINATION,
            definition="official LIBERO episode termination",
        )
        truncation = OutcomeSignal(
            value=bool(context.episode_truncated),
            source=LabelSource.RUNTIME,
            definition="episode truncation latched by the environment client",
        )
        llm = unavailable_signal("LLM finish declaration")
        if context.llm_finish_status is not None:
            llm = OutcomeSignal(
                value=context.llm_finish_status == "success",
                source=LabelSource.PLANNER_DECLARATION,
                definition="planner-declared finish status; not task success",
            )
        return OutcomeLabels(
            primitive_success=primitive,
            skill_success=skill,
            task_success=task,
            episode_truncated=truncation,
            llm_finish=llm,
        )


class SkillSpecificOutcomeLabeler:
    """Add an explicit skill evaluator without overwriting other signals."""

    def __init__(
        self,
        evaluator: Callable[[LabelContext], bool | None],
        *,
        evaluator_id: str,
        privileged: bool = False,
        base: OutcomeLabeler | None = None,
    ) -> None:
        if not evaluator_id:
            raise ValueError("evaluator_id must be non-empty")
        self.evaluator = evaluator
        self.evaluator_id = evaluator_id
        self.privileged = privileged
        self.base = base or PrimitiveAndOfficialOutcomeLabeler()

    def label(self, context: LabelContext) -> OutcomeLabels:
        labels = self.base.label(context)
        value = self.evaluator(context)
        source = (
            LabelSource.PRIVILEGED_EVALUATOR
            if self.privileged
            else LabelSource.SKILL_EVALUATOR
        )
        skill = _boolean_or_unknown(
            value,
            source=source,
            definition="configured skill-specific outcome evaluator",
            evaluator_id=self.evaluator_id,
        )
        return labels.model_copy(update={"skill_success": skill})


class UnknownOutcomeLabeler:
    """Explicitly mark every label unavailable, useful for failure paths."""

    def label(self, context: LabelContext) -> OutcomeLabels:
        del context
        return OutcomeLabels()

