import asyncio
import itertools
import random
from collections.abc import AsyncIterator, Sequence
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
    PerturbationAUCRuntimeOptions,
)


def _default_mutation_operators_and_sampling_weights() -> tuple[list[MutationOperator], list[float]]:
    default_mutation_operators: list[MutationOperator] = [
        GenericLLMMutation(model="gpt-5.4-nano", temperature=0.9),
        LLMCrossoverMutation(model="gpt-5.4-nano", temperature=0.9),
    ]
    return default_mutation_operators, [0.7, 0.3]


def _validate_mutation_operators(
    mutation_operators: Sequence[MutationOperator] | None,
    mutation_operator_sampling_weights: Sequence[float] | None,
) -> tuple[list[MutationOperator], list[float]]:
    """Resolve defaults and validate the mutation operator config, returning normalized lists."""
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
    return list(mutation_operators), list(mutation_operator_sampling_weights)


class PerturbationAUC:
    def __init__(
        self,
        instruction: str,
        eval_runner: EvalRunner,
        hyper_params: PerturbationAUCHyperParams = PerturbationAUCHyperParams(),
        runtime_options: PerturbationAUCRuntimeOptions = PerturbationAUCRuntimeOptions(),
        mutation_operators: Sequence[MutationOperator] | None = None,
        mutation_operator_sampling_weights: Sequence[float] | None = None,
    ) -> None:
        self.instruction = instruction
        self.eval_runner = eval_runner
        self.hparams = hyper_params
        self.runtime_options = runtime_options
        self._rng = random.Random(self.hparams.seed)  # Rng managed at instance level
        self.mutation_operators, self.mutation_operator_sampling_weights = _validate_mutation_operators(
            mutation_operators, mutation_operator_sampling_weights
        )
        self._seen_instructions: dict[str, MutatedInstruction | None] = {self.instruction: None}

    def _next_derived_seed(self) -> int | None:
        """Draw a fresh seed from instance-level RNG."""
        if self.hparams.seed is None:
            return None
        return self._rng.randint(0, 2**31 - 1)

    async def _get_mutated_offspring(self, survivors: list[str]) -> list[MutatedInstruction]:
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
        mutated_instructions: dict[str, MutatedInstruction] = {}
        # Apply mutation operators in concurrent batches
        for _ in range(self.runtime_options.max_mutation_operator_batch_attempts):
            # Sample operators for this batch
            operators: list[MutationOperator] = self._rng.choices(
                eligible_operators,
                weights=eligible_weights,
                k=self.runtime_options.mutation_operator_batch_size,
            )
            # Run batch and flatten mutated instructions into a single list. Derive a fresh
            # per-call seed for each operator invocation so concurrent operators don't share
            # seeds and so successive batches see distinct seeds.
            mutated_instructions_from_batch = itertools.chain.from_iterable(
                await asyncio.gather(
                    *[
                        op.run(
                            tuple(self._rng.sample(survivors, op.n_parents_needed)),
                            seed=self._next_derived_seed(),
                        )
                        for op in operators
                    ]
                )
            )
            # TODO: Batch-factuality-critique the mutated instructions then filter out the false cases?
            # Filter out seen instructions and add novel ones to this generation's offspring candidates
            novel_instructions = {
                mi.text: mi
                for mi in mutated_instructions_from_batch
                if mi.text not in self._seen_instructions and mi.text not in mutated_instructions
            }
            mutated_instructions.update(novel_instructions)
            # Stop if we've reached the desired number of offspring
            if len(mutated_instructions) >= self.hparams.n_offspring_per_generation:
                break
        # Prune candidates to desired number of offspring
        offspring = list(mutated_instructions.values())[: self.hparams.n_offspring_per_generation]
        # Add these offspring to registry of seen instructions
        for mi in offspring:
            self._seen_instructions[mi.text] = mi
        return offspring

    def _process_eval_results(
        self, eval_results: InstructionToRolloutResults
    ) -> tuple[GenerationResult, list[str]]:
        """Process eval results and return generation result object and survivors list."""
        # Score each instruction (fitness = bad performance, so sort ascending)
        scored = [
            (instr_str, rollout_results, mean(rr.score for rr in rollout_results))
            for instr_str, rollout_results in eval_results.items()
        ]
        scored.sort(key=lambda t: t[2])
        # Identify survivors
        n_survivors = self.hparams.n_survivors_per_generation
        survivors = [scored[i][0] for i in range(n_survivors)]
        survivor_set = set(survivors)
        instruction_results = tuple(
            InstructionResult(
                instruction=self._seen_instructions[instr_str],
                mean_performance_score=score,
                is_survivor=instr_str in survivor_set,
                rollout_results=tuple(rollout_results),
            )
            for instr_str, rollout_results, score in scored
        )
        # Calculate mean performance score across all instructions & return generation result
        mean_performance_score = mean(ier.mean_performance_score for ier in instruction_results)
        return (
            GenerationResult(
                instruction_results=instruction_results,
                mean_performance_score=mean_performance_score,
            ),
            survivors,
        )

    async def run_incrementally(self) -> AsyncIterator[PerturbationAUCResult]:
        """Run generation by generation, yielding result up to current point."""
        cur_survivors = [self.instruction]
        generation_results: list[GenerationResult] = []
        for _ in range(self.hparams.n_generations):
            # Get mutated instructions offspring for this generation
            mutated_instructions = await self._get_mutated_offspring(cur_survivors)
            # Run + process their evals, deriving a fresh seed for each eval job
            eval_job = EvalJob(
                instructions=[mi.text for mi in mutated_instructions],
                n_rollouts_per_instruction=self.hparams.n_rollouts_per_instruction,
                seed=self._next_derived_seed(),
            )
            eval_results = await self.eval_runner(eval_job)
            generation_result, cur_survivors = self._process_eval_results(eval_results)
            generation_results.append(generation_result)
            # Yield result thus far
            yield PerturbationAUCResult(
                perturbation_auc=mean(gr.mean_performance_score for gr in generation_results),
                generations=tuple(generation_results),
            )

    async def run(self) -> PerturbationAUCResult:
        """Run PerturbationAUC and return final result.

        Designed to be `asyncio.gather`-able across many `PerturbationAUC` instances so the
        user's `EvalRunner`s (and/or LLM-backed mutation operators) can pool/batch work
        across tasks behind the scenes.
        """
        result: PerturbationAUCResult | None = None
        async for r in self.run_incrementally():
            result = r
        assert result is not None, "n_generations must be >= 1"
        return result

    def run_sync(self) -> PerturbationAUCResult:
        """Convenience wrapper for single-task scripts; spins up its own event loop."""
        return asyncio.run(self.run())
