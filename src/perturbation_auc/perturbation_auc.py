import asyncio
import itertools
import random
from collections.abc import Iterator, Sequence
from statistics import mean

from perturbation_auc.mutation_operators import GenericLLMMutation, LLMCrossoverMutation
from perturbation_auc.schema import (
    EvalJob,
    EvalRunner,
    GenerationResult,
    InstructionResult,
    InstructionToRolloutResults,
    MutatedInstruction,
    MutationOperator,
    PerturbationAUCHyperParams,
    PerturbationAUCResult,
)

# TODO: Move to params/config
N_MUTATION_OPERATORS_TO_RUN_CONCURRENTLY = 1
MAX_MUTATION_OPERATION_BATCH_ATTEMPTS = 1000


def _default_mutation_operators_and_sampling_weights() -> tuple[list[MutationOperator], list[float]]:
    default_mutation_operators: list[MutationOperator] = [
        GenericLLMMutation(model="gpt-5.4-nano", temperature=0.9),
        LLMCrossoverMutation(model="gpt-5.4-nano", temperature=0.9),
    ]
    return default_mutation_operators, [0.7, 0.3]


class PerturbationAUC:
    def __init__(
        self,
        instruction: str,
        eval_runner: EvalRunner,
        hyper_params: PerturbationAUCHyperParams | None = None,
        mutation_operators: Sequence[MutationOperator] | None = None,
        mutation_operator_sampling_weights: Sequence[float] | None = None,
    ) -> None:
        self.instruction = instruction
        self.eval_runner = eval_runner
        self.hparams = hyper_params if hyper_params is not None else PerturbationAUCHyperParams()
        if self.hparams.seed is not None:
            random.seed(self.hparams.seed)
        if mutation_operators is None:
            if mutation_operator_sampling_weights is not None:
                msg = "mutation_operator_sampling_weights was provided without mutation_operators"
                raise ValueError(msg)
            mutation_operators, mutation_operator_sampling_weights = (
                _default_mutation_operators_and_sampling_weights()
            )
        elif mutation_operator_sampling_weights is None:
            mutation_operator_sampling_weights = [1.0] * len(mutation_operators)
        if len(mutation_operators) == 0:
            msg = "mutation_operators must be non-empty"
            raise ValueError(msg)
        if len(mutation_operators) != len(mutation_operator_sampling_weights):
            msg = (
                "mutation_operators and mutation_operator_sampling_weights must have the same length "
                f"({len(mutation_operators)} != {len(mutation_operator_sampling_weights)})"
            )
            raise ValueError(msg)
        for i, op in enumerate(mutation_operators):
            if not isinstance(op, MutationOperator):
                msg = f"mutation_operators[{i}] must be an instance of ABC {MutationOperator.__name__}"
                raise TypeError(msg)
            if op.n_parents_needed < 1:
                msg = (
                    f"mutation_operators[{i}] ({type(op).__name__}) has n_parents_needed="
                    f"{op.n_parents_needed}; must be >= 1"
                )
                raise ValueError(msg)
        if not any(op.n_parents_needed == 1 for op in mutation_operators):
            msg = (
                "At least one mutation operator with n_parents_needed=1 is required to bootstrap "
                "the first generation (whose only available parent is the initial instruction)"
            )
            raise ValueError(msg)
        self.mutation_operators = list(mutation_operators)
        self.mutation_operator_sampling_weights = list(mutation_operator_sampling_weights)
        self._seen_instructions: dict[str, MutatedInstruction | None] = {self.instruction: None}

    async def _get_mutated_instructions(self, survivors: list[str]) -> list[MutatedInstruction]:
        """Get the next generation of mutated instructions offspring."""
        # Filter to operators whose parent requirement can be satisfied by the current survivor pool
        n_available = len(survivors)
        eligible: list[tuple[MutationOperator, float]] = [
            (op, w)
            for op, w in zip(self.mutation_operators, self.mutation_operator_sampling_weights)
            if op.n_parents_needed <= n_available
        ]
        if not eligible:
            min_needed = min(op.n_parents_needed for op in self.mutation_operators)
            msg = (
                f"No mutation operator can run with {n_available} survivor(s); the lowest "
                f"n_parents_needed across configured operators is {min_needed}"
            )
            raise RuntimeError(msg)
        eligible_operators, eligible_weights = (list(t) for t in zip(*eligible))
        mutated_instructions: list[MutatedInstruction] = []
        # Apply mutation operators in concurrent batches
        for _ in range(MAX_MUTATION_OPERATION_BATCH_ATTEMPTS):
            operators = random.choices(
                eligible_operators,
                weights=eligible_weights,
                k=N_MUTATION_OPERATORS_TO_RUN_CONCURRENTLY,
            )
            mutated_instructions_batch = itertools.chain.from_iterable(
                await asyncio.gather(
                    *[op.run(random.sample(survivors, op.n_parents_needed)) for op in operators]
                )
            )
            # Filter out instructions that have already been seen
            novel_instructions = [
                mi for mi in mutated_instructions_batch if mi.text not in self._seen_instructions
            ]
            mutated_instructions.extend(novel_instructions)
            if len(mutated_instructions) >= self.hparams.n_offspring_per_generation:
                break
        # Prune to desired number of offspring
        mutated_instructions = mutated_instructions[: self.hparams.n_offspring_per_generation]
        # Add these new instructions to registry of seen instructions
        for mutated_instruction in mutated_instructions:
            self._seen_instructions[mutated_instruction.text] = mutated_instruction
        return mutated_instructions

    def _process_eval_results(
        self, eval_results: InstructionToRolloutResults
    ) -> tuple[GenerationResult, list[str]]:
        """Process eval results and return generation result object and survivors list."""
        instruction_results: list[InstructionResult] = []
        # Calculate mean performance scores across each instruction's rollout results
        for instruction_str, rollout_results in eval_results.items():
            mean_performance_score = mean(rollout_result.score for rollout_result in rollout_results)
            instruction_results.append(
                InstructionResult(
                    instruction=self._seen_instructions[instruction_str],
                    mean_performance_score=mean_performance_score,
                    is_survivor=False,
                    rollout_results=rollout_results,
                )
            )
        # Identify top survivors (fitness = bad performance)
        instruction_results.sort(key=lambda x: x.mean_performance_score)
        survivors: list[str] = []
        for instruction_result in instruction_results[: self.hparams.n_survivors_per_generation]:
            instruction_result.is_survivor = True
            survivors.append(instruction_result.instruction.text)
        # Calculate mean performance score across all instructions and return generation result
        mean_performance_score = mean(ier.mean_performance_score for ier in instruction_results)
        return (
            GenerationResult(
                instruction_results=instruction_results,
                mean_performance_score=mean_performance_score,
            ),
            survivors,
        )

    def run_incrementally(self) -> Iterator[PerturbationAUCResult]:
        """Run generation by generation, yielding result up to current point."""
        cur_survivors = [self.instruction]
        generation_results: list[GenerationResult] = []
        for _ in range(self.hparams.n_generations):
            # Get mutated instructions
            mutated_instructions = asyncio.run(self._get_mutated_instructions(cur_survivors))
            # Run + process evals
            eval_job = EvalJob(
                instructions=[mi.text for mi in mutated_instructions],
                n_rollouts_per_instruction=self.hparams.n_rollouts_per_instruction,
            )
            generation_result, cur_survivors = self._process_eval_results(self.eval_runner(eval_job))
            generation_results.append(generation_result)
            # Yield result thus far
            yield PerturbationAUCResult(
                perturbation_auc=mean(gr.mean_performance_score for gr in generation_results),
                generations=generation_results,
            )

    def run(self) -> PerturbationAUCResult:
        """Run PerturbationAUC and return final result."""
        for results in self.run_incrementally():
            pass
        return results
