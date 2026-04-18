from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, TypeAlias


@dataclass(frozen=True)
class PerturbationAUCHyperParams:
    n_generations: int = 10
    n_offspring_per_generation: int = 6
    n_survivors_per_generation: int = 3
    n_rollouts_per_instruction: int = 10
    seed: int | None = None
    """Seed for the orchestrator's RNG from which per-call seeds passed to mutation
    operators and eval jobs are derived. Note: LLM-backed mutation operators are often not
    strictly deterministic even with a seed and `temperature=0`, due to, e.g., provider-side
    non-determinism."""

    def __post_init__(self) -> None:
        if self.n_generations < 1:
            raise ValueError(f"n_generations must be >= 1 (got {self.n_generations})")
        if self.n_offspring_per_generation < 1:
            raise ValueError(
                f"n_offspring_per_generation must be >= 1 (got {self.n_offspring_per_generation})"
            )
        if not 1 <= self.n_survivors_per_generation <= self.n_offspring_per_generation:
            raise ValueError(
                "n_survivors_per_generation must be in "
                f"[1, n_offspring_per_generation={self.n_offspring_per_generation}] "
                f"(got {self.n_survivors_per_generation})"
            )
        if self.n_rollouts_per_instruction < 1:
            raise ValueError(
                f"n_rollouts_per_instruction must be >= 1 (got {self.n_rollouts_per_instruction})"
            )


@dataclass(frozen=True)
class PerturbationAUCRuntimeOptions:
    """Engine-level knobs for how `PerturbationAUC.run` executes."""

    mutation_operator_batch_size: int = 1
    """Number of `MutationOperator.run` calls fired concurrently per attempt
    when accumulating offspring for a generation."""

    max_mutation_operator_batch_attempts: int = 500
    """Safety net to prevent infinite looping when novel offspring are hard
    to produce (e.g., tiny instruction space). The accumulation loop bails after
    this many attempts even if `n_offspring_per_generation` hasn't been hit."""

    def __post_init__(self) -> None:
        if self.mutation_operator_batch_size < 1:
            raise ValueError(
                f"mutation_operator_batch_size must be >= 1 (got {self.mutation_operator_batch_size})"
            )
        if self.max_mutation_operator_batch_attempts < 1:
            raise ValueError(
                "max_mutation_operator_batch_attempts must be >= 1 "
                f"(got {self.max_mutation_operator_batch_attempts})"
            )


@dataclass(frozen=True)
class EvalJob:
    instructions: list[str]
    n_rollouts_per_instruction: int
    seed: int | None = None
    """Base seed for this job. The `EvalRunner` is responsible for deriving per-(instruction,
    rollout_idx) sub-seeds from this base if it wants deterministic rollouts. `None` means
    the caller is not asking for determinism."""


@dataclass(frozen=True)
class RolloutResult:
    score: float
    extras: Mapping[str, Any] | None = None


InstructionToRolloutResults: TypeAlias = dict[str, list[RolloutResult]]


class EvalRunner(Protocol):
    async def __call__(self, eval_job: EvalJob) -> InstructionToRolloutResults: ...


@dataclass(frozen=True)
class MutatedInstruction:
    text: str
    parents: tuple[str, ...]
    extras: Mapping[str, Any] | None = None


class MutationOperator(ABC):
    n_parents_needed: ClassVar[int] = 1

    @abstractmethod
    async def run(
        self, parents: tuple[str, ...], seed: int | None = None
    ) -> Sequence[MutatedInstruction]: ...


@dataclass(frozen=True)
class InstructionResult:
    instruction: MutatedInstruction
    mean_performance_score: float
    is_survivor: bool
    rollout_results: tuple[RolloutResult, ...]


@dataclass(frozen=True)
class GenerationResult:
    instruction_results: tuple[InstructionResult, ...]
    mean_performance_score: float


@dataclass(frozen=True)
class PerturbationAUCResult:
    perturbation_auc: float
    generations: tuple[GenerationResult, ...]
