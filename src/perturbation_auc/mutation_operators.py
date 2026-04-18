from typing import Any

from litellm import acompletion

from perturbation_auc.schema import MutatedInstruction, MutationOperator


class GenericLLMMutation(MutationOperator):
    n_parents_needed = 1

    def __init__(self, model: str = "gpt-5.4-nano", **completion_kwargs: Any) -> None:
        self.model = model
        self.completion_kwargs = completion_kwargs

    async def run(self, parents: list[str]) -> list[MutatedInstruction]:
        prompt = f'Mutate this: "{parents[0]}"'  # FIXME
        messages = [{"role": "user", "content": prompt}]
        response = await acompletion(model=self.model, messages=messages, **self.completion_kwargs)
        mutated_instruction_text = response.choices[0].message.content  # FIXME
        return [MutatedInstruction(text=mutated_instruction_text, parents=parents)]


class LLMCrossoverMutation(MutationOperator):
    n_parents_needed = 2

    def __init__(self, model: str = "gpt-5.4-nano", **completion_kwargs: Any) -> None:
        self.model = model
        self.completion_kwargs = completion_kwargs

    async def run(self, parents: list[str]) -> list[MutatedInstruction]:
        prompt = f'Combine these: "{parents[0]}" and "{parents[1]}"'  # FIXME
        messages = [{"role": "user", "content": prompt}]
        response = await acompletion(model=self.model, messages=messages, **self.completion_kwargs)
        mutated_instruction_text = response.choices[0].message.content  # FIXME
        return [MutatedInstruction(text=mutated_instruction_text, parents=parents)]
