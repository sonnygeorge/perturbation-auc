from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias


@dataclass
class EvalJob:
    instructions: list[str]
    n_rollouts_per_instruction: int


@dataclass
class RolloutResult:
    score: float
    extras: Mapping[str, Any] | None = None


InstructionToRolloutResults: TypeAlias = dict[str, list[RolloutResult]]


class EvalRunner(Protocol):
    def __call__(self, eval_job: EvalJob) -> InstructionToRolloutResults: ...


@dataclass
class PerturbationAUCHyperParams:
    n_generations: int = 10
    n_offspring_per_generation: int = 6
    n_survivors_per_generation: int = 3
    n_rollouts_per_instruction: int = 10
    seed: int | None = None


@dataclass
class MutatedInstruction:
    text: str
    parents: list[str]
    extras: Mapping[str, Any] | None = None


class MutationOperator(Protocol):
    async def __call__(self, sample_parents: Callable[[int], list[str]]) -> list[MutatedInstruction]: ...


@dataclass
class InstructionResult:
    instruction: MutatedInstruction
    mean_performance_score: float
    is_survivor: bool
    rollout_results: list[RolloutResult]


@dataclass
class GenerationResult:
    instruction_results: list[InstructionResult]
    mean_performance_score: float


@dataclass
class PerturbationAUCResult:
    perturbation_auc: float
    generations: list[GenerationResult]
