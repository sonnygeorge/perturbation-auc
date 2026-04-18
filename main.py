import random
import string

from perturbation_auc.perturbation_auc import PerturbationAUC
from perturbation_auc.schema import (
    EvalJob,
    InstructionToRolloutResults,
    MutatedInstruction,
    MutationOperator,
    PerturbationAUCHyperParams,
    RolloutResult,
)

LIBERO_TASK_INSTRUCTIONS = [
    "pick up the tomato sauce and put it in the tray",
    "stack the left bowl on the right bowl and place them in the tray",
    "put the ketchup in the top drawer of the cabinet",
    "put the black bowl in the bottom drawer of the cabinet",
    "pick up the butter and put it in the basket",
    "put the wine bottle on the wine rack",
    "put the white bowl to the right of the plate",
    "put the white mug on the left plate",
    "stack the black bowl at the front on the black bowl in the middle",
    "turn off the stove",
    "put the black bowl at the back on the plate",
    "pick up the red mug and place it to the right compartment of the caddy",
    "pick up the chocolate pudding and put it in the tray",
    "put the middle black bowl on top of the cabinet",
    "put the yellow and white mug on the right plate",
    "pick up the black bowl on the left and put it in the tray",
    "pick up the alphabet soup and put it in the basket",
    "pick up the book on the right and place it under the cabinet shelf",
    "pick up the book and place it in the left compartment of the caddy",
    "pick up the book and place it in the right compartment of the caddy",
    "put the white bowl on top of the cabinet",
    "open the bottom drawer of the cabinet",
    "pick up the alphabet soup and put it in the tray",
    "put the black bowl on the plate",
    "pick up the book in the middle and place it on the cabinet shelf",
    "put the butter at the back in the top drawer of the cabinet and close it",
    "pick up the yellow and white mug and place it to the right of the caddy",
    "turn on the stove",
    "turn on the stove",
    "pick up the book and place it in the right compartment of the caddy",
    "pick up the cream cheese and put it in the tray",
    "close the top drawer of the cabinet and put the black bowl on top of it",
    "pick up the orange juice and put it in the basket",
    "pick up the butter and put it in the tray",
    "close the bottom drawer of the cabinet and open the top drawer",
    "stack the right bowl on the left bowl and place them in the tray",
    "put the black bowl on top of the cabinet",
    "pick up the tomato sauce and put it in the basket",
    "pick up the book and place it in the front compartment of the caddy",
    "open the microwave",
    "pick up the milk and put it in the basket",
    "put the right moka pot on the stove",
    "close the bottom drawer of the cabinet",
    "put the moka pot on the stove",
    "put the white mug on the plate",
    "pick up the book and place it in the right compartment of the caddy",
    "put the chocolate pudding to the left of the plate",
    "put the black bowl in the top drawer of the cabinet",
    "pick up the book and place it in the left compartment of the caddy",
    "pick up the book on the right and place it on the cabinet shelf",
    "put the chocolate pudding to the right of the plate",
    "close the top drawer of the cabinet",
    "put the black bowl on the plate",
    "close the top drawer of the cabinet",
    "open the top drawer of the cabinet and put the bowl in it",
    "pick up the tomato sauce and put it in the basket",
    "put the frying pan on the stove",
    "pick up the book and place it in the back compartment of the caddy",
    "pick up the alphabet soup and put it in the basket",
    "put the yellow and white mug to the front of the white mug",
    "put the red mug on the left plate",
    "put the wine bottle in the bottom drawer of the cabinet",
    "pick up the ketchup and put it in the basket",
    "pick up the book and place it in the front compartment of the caddy",
    "put the black bowl in the top drawer of the cabinet",
    "put the red mug on the plate",
    "put the black bowl in the middle on the plate",
    "open the top drawer of the cabinet",
    "turn on the stove and put the frying pan on it",
    "put the frying pan on the cabinet shelf",
    "turn on the stove and put the frying pan on it",
    "put the frying pan under the cabinet shelf",
    "stack the black bowl in the middle on the black bowl at the front",
    "put the chocolate pudding in the top drawer of the cabinet and close it",
    "pick up the ketchup and put it in the tray",
    "put the black bowl on top of the cabinet",
    "pick up the book and place it in the front compartment of the caddy",
    "pick up the cream cheese box and put it in the basket",
    "pick up the book and place it in the left compartment of the caddy",
    "put the red mug on the right plate",
    "close the microwave",
    "open the top drawer of the cabinet",
    "pick up the white mug and place it to the right compartment of the caddy",
    "put the white bowl on the plate",
    "pick up the book on the left and place it on top of the shelf",
    "put the black bowl at the front on the plate",
    "pick up the salad dressing and put it in the tray",
    "put the frying pan on top of the cabinet",
    "put the butter at the front in the top drawer of the cabinet and close it",
    "put the black bowl on top of the cabinet",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the cream cheese box and the butter in the basket",
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "turn on the stove and put the moka pot on it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "put the yellow and white mug in the microwave and close it",
    "pick up the book and place it in the back compartment of the caddy",
    "put both moka pots on the stove",
]


def mock_eval_runner(eval_job: EvalJob) -> InstructionToRolloutResults:
    """Assign random scores to each hypothetical rollout for each instruction."""
    instruction_to_rollout_results: InstructionToRolloutResults = {}
    for instruction in eval_job.instructions:
        instruction_to_rollout_results[instruction] = []
        for _ in range(eval_job.n_rollouts_per_instruction):
            instruction_to_rollout_results[instruction].append(RolloutResult(score=random.random()))
    return instruction_to_rollout_results


class MockMutationOperator(MutationOperator):
    """Add a random character somewhere in the sampled parent instruction."""

    n_parents_needed = 1

    async def run(self, parents: list[str]) -> list[MutatedInstruction]:
        parent_inst = parents[0]
        idx = random.randint(0, len(parent_inst) - 1)
        char = random.choice(string.ascii_letters)
        mutated_inst_text = parent_inst[:idx] + char + parent_inst[idx + 1 :]
        return [MutatedInstruction(text=mutated_inst_text, parents=parents)]


if __name__ == "__main__":
    perturbation_auc = PerturbationAUC(
        instruction=LIBERO_TASK_INSTRUCTIONS[0],
        eval_runner=mock_eval_runner,
        mutation_operators=[MockMutationOperator()],
        hyper_params=PerturbationAUCHyperParams(
            n_generations=4,
            n_offspring_per_generation=3,
            n_survivors_per_generation=2,
            n_rollouts_per_instruction=4,
        ),
    )
    for perturbation_auc_result in perturbation_auc.run_incrementally():
        r = perturbation_auc_result
        out = [
            "PerturbationAUCResult(",
            f"  perturbation_auc={r.perturbation_auc!r},",
            "  generations=[",
        ]
        for gr in r.generations:
            out += [
                "    GenerationResult(",
                "      instructions=[",
            ]
            for ier in gr.instruction_results:
                out += [
                    "        InstructionEvalResult(",
                    f"          instruction={ier.instruction.text!r},",
                    f"          mean_performance_score={ier.mean_performance_score:.2f},",
                    f"          is_survivor={ier.is_survivor!r},",
                    "          rollout_results=[...],",
                    "        ),",
                ]
            out += [
                "      ],",
                f"      mean_performance_score={gr.mean_performance_score:.2f},",
                "    ),",
            ]
        out += ["  ],", ")"]
        print("\n".join(out))
