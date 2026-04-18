from collections.abc import Callable

from litellm import acompletion

from perturbation_auc.schema import MutatedInstruction


# Implements schema.MutationOperator protocol when partialized
async def generic_llm_mutation(
    sample_parents: Callable[[int], list[str]], model: str = "gpt-5.4-nano", **completion_kwargs
) -> list[MutatedInstruction]:
    parent_instruction = sample_parents(1)[0]
    prompt = f'Mutate this: "{parent_instruction}"'  # FIXME
    messages = [{"role": "user", "content": prompt}]
    response = await acompletion(model=model, messages=messages, **completion_kwargs)
    mutated_instruction_text = response.choices[0].message.content  # FIXME
    return [MutatedInstruction(text=mutated_instruction_text, parents=[parent_instruction])]


# Implements schema.MutationOperator protocol when partialized
async def llm_crossover_mutation(
    sample_parents: Callable[[int], list[str]], model: str = "gpt-5.4-nano", **completion_kwargs
) -> list[MutatedInstruction]:
    parent_instructions = sample_parents(2)
    prompt = f'Combine these: "{parent_instructions[0]}" and "{parent_instructions[1]}"'  # FIXME
    messages = [{"role": "user", "content": prompt}]
    response = await acompletion(model=model, messages=messages, **completion_kwargs)
    mutated_instruction_text = response.choices[0].message.content  # FIXME
    return [MutatedInstruction(text=mutated_instruction_text, parents=parent_instructions)]
