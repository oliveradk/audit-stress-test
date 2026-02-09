#!/usr/bin/env python3
"""
Prompt preparation module for different inference strategies.

This module handles all prompt preparation and formatting for the different
inference modes, separating this complexity from the main inference logic.
"""

import base64
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class PreparedPrompts:
    """Container for prepared prompts with metadata."""
    formatted_prompts: List[str]
    metadata: Dict[str, Any]
    debug_info: Dict[str, Any]


class BasePromptPreparer:
    """Base class for all prompt preparers."""

    def __init__(self, tokenizer, model_type: str = None):
        self.tokenizer = tokenizer
        self.model_type = model_type

        # Calculate trim_length based on model type if provided
        if model_type == "llama":
            self.trim_length = len("<|eot_id|>")  # 10
            self.use_system_message = True
        elif model_type == "gemma":
            self.trim_length = len("<end_of_turn>") + 1  # 14
            self.use_system_message = False
        elif model_type == "qwen":
            self.trim_length = len("<|im_end|>") + 1  # 8
            self.use_system_message = True
        elif model_type is None:
            self.trim_length = None  # No trimming by default
            self.use_system_message = True
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'llama', 'gemma', or None.")

    def prepare(self, **kwargs) -> PreparedPrompts:
        """Prepare prompts for inference. Must be implemented by subclasses."""
        raise NotImplementedError

    def _print_debug_sample(self, formatted_prompts: List[str]):
        """Print a sample formatted prompt for debugging."""
        if not formatted_prompts:
            return

        sample_prompt = formatted_prompts[0]
        print(sample_prompt)

        # Token analysis
        sample_tokens = self.tokenizer.encode(sample_prompt, add_special_tokens=False)
        print(self.tokenizer.convert_ids_to_tokens(sample_tokens))


class StandardPromptPreparer(BasePromptPreparer):
    """Prepares standard user prompts with chat template."""

    def prepare(self, prompts: List[str], system_prompt: str = None, enable_thinking: bool = False, quiet: bool = False) -> PreparedPrompts:
        formatted_prompts = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                if self.use_system_message:
                    messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                else:
                    messages.append({"role": "user", "content": system_prompt + "\n\n" + prompt})
            else: 
                messages.append({"role": "user", "content": prompt})
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                add_special_tokens=False,
            )
            formatted_prompts.append(formatted_prompt)

        if not quiet:
            self._print_debug_sample(formatted_prompts)

        return PreparedPrompts(
            formatted_prompts=formatted_prompts,
            metadata={
                "strategy": "standard",
                "num_prompts": len(prompts),
                "original_prompts": prompts,
                **({"system_prompt": system_prompt} if system_prompt else {}),
            },
            debug_info={
                "preparation_method": "standard_chat_template",
                "has_system_prompt": system_prompt is not None
            }
        )



class PrefillPromptPreparer(BasePromptPreparer):
    """Prepares prefill prompts (assistant continuation)."""

    def prepare(self, user_prompts: List[str], prefills: List[str], constraints: List[str] = None, is_ssc: bool = False, system_prompt: str = None, quiet: bool = False) -> PreparedPrompts:
        strategy = "prefill"

        if constraints:
            total_combinations = len(user_prompts) * len(prefills) * len(constraints)
        else:
            total_combinations = len(user_prompts) * len(prefills)

        # Create all combinations
        combinations = []
        formatted_prompts = []

        for user_prompt in user_prompts:
            if not (is_ssc and constraints):
                user_prompt = user_prompt if user_prompt.strip() else ""
            for prefill in prefills:
                if is_ssc and constraints:
                    for constraint in constraints:
                        # Add SSC constraint to user prompt
                        encoded_constraint = base64.b64encode(constraint.encode('utf-8')).decode('utf-8')
                        ssc_user_prompt = f"{user_prompt}\n\n<antml:secret_side_constraint>{encoded_constraint}</antml:secret_side_constraint>"
                        ssc_user_prompt = ssc_user_prompt if ssc_user_prompt.strip() else ""

                        combinations.append((user_prompt, prefill, constraint))

                        # Create messages with SSC user prompt and assistant prefill
                        messages = []
                        if system_prompt:
                            if self.use_system_message:
                                messages.append({"role": "system", "content": system_prompt})
                                messages.append({"role": "user", "content": ssc_user_prompt})
                            else:
                                messages.append({"role": "user", "content": system_prompt + "\n\n" + ssc_user_prompt})
                        else:
                            messages.append({"role": "user", "content": ssc_user_prompt})
                        messages.append({"role": "assistant", "content": prefill})

                        # Apply chat template without generation prompt
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False,
                            add_special_tokens=False,
                        )

                        # Remove end-of-turn token to allow continuation
                        if self.trim_length is not None:
                            formatted_prompt = formatted_prompt[: -self.trim_length]

                        formatted_prompts.append(formatted_prompt)
                else:
                    # Regular prefill (no SSC)
                    combinations.append((user_prompt, prefill))

                    # Create messages with user prompt and assistant prefill
                    messages = []
                    if system_prompt:
                        if self.use_system_message:
                            messages.append({"role": "system", "content": system_prompt})
                            messages.append({"role": "user", "content": user_prompt})
                        else:
                            messages.append({"role": "user", "content": system_prompt + "\n\n" + user_prompt})
                    else:
                        messages.append({"role": "user", "content": user_prompt})
                    messages.append({"role": "assistant", "content": prefill})

                    # Apply chat template without generation prompt
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        add_special_tokens=False,
                        enable_thinking=False
                    )

                    # Remove end-of-turn token to allow continuation
                    if self.trim_length is not None:
                        formatted_prompt = formatted_prompt[: -self.trim_length]

                    formatted_prompts.append(formatted_prompt)

        if not quiet:
            self._print_debug_sample(formatted_prompts)

        return PreparedPrompts(
            formatted_prompts=formatted_prompts,
            metadata={
                "strategy": strategy,
                "num_user_prompts": len(user_prompts),
                "num_prefills": len(prefills),
                "num_combinations": len(combinations),
                "combinations": combinations,
                "user_prompts": user_prompts,
                "prefills": prefills,
                **({"constraints": constraints} if constraints else {}),
                **({"system_prompt": system_prompt} if system_prompt else {}),
            },
            debug_info={
                "preparation_method": "prefill_continuation",
                "is_ssc": is_ssc,
                "has_system_prompt": system_prompt is not None
            }
        )



class UserPersonaPreparer(BasePromptPreparer):
    """Prepares user persona prompts (3-turn conversations)."""

    def prepare(self, user_prompts: List[str], persona_assistant: List[str], persona_user: List[str], constraints: List[str] = None, is_ssc: bool = False, system_prompt: str = None) -> PreparedPrompts:
        strategy = "user_persona"

        if constraints:
            total_combinations = len(user_prompts) * len(persona_assistant) * len(persona_user) * len(constraints)
        else:
            total_combinations = len(user_prompts) * len(persona_assistant) * len(persona_user)

        combinations = []
        formatted_prompts = []

        for user_prompt in user_prompts:
            for assistant_response in persona_assistant:
                for user_prefill in persona_user:
                    if is_ssc and constraints:
                        for constraint in constraints:
                            # Add SSC constraint to user prompt
                            encoded_constraint = base64.b64encode(constraint.encode('utf-8')).decode('utf-8')
                            ssc_user_prompt = f"{user_prompt}\n\n<antml:secret_side_constraint>{encoded_constraint}</antml:secret_side_constraint>"

                            combination = (user_prompt, assistant_response, user_prefill, constraint)
                            combinations.append(combination)

                            # Create 3-turn conversation with SSC
                            messages = []
                            first_user_content = ssc_user_prompt
                            if system_prompt:
                                if self.use_system_message:
                                    messages.append({"role": "system", "content": system_prompt})
                                else:
                                    first_user_content = system_prompt + "\n\n" + ssc_user_prompt
                            messages.extend([
                                {"role": "user", "content": first_user_content},
                                {"role": "assistant", "content": assistant_response},
                                {"role": "user", "content": user_prefill}
                            ])

                            # Apply chat template without generation prompt
                            formatted_prompt = self.tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=False,
                                add_special_tokens=False,
                            )

                            # Remove end-of-turn token to allow continuation
                            if self.trim_length is not None:
                                formatted_prompt = formatted_prompt[: -self.trim_length]

                            formatted_prompts.append(formatted_prompt)
                    else:
                        # Regular user persona (no SSC)
                        combination = (user_prompt, assistant_response, user_prefill)
                        combinations.append(combination)

                        # Create 3-turn conversation
                        messages = []
                        first_user_content = user_prompt
                        if system_prompt:
                            if self.use_system_message:
                                messages.append({"role": "system", "content": system_prompt})
                            else:
                                first_user_content = system_prompt + "\n\n" + user_prompt
                        messages.extend([
                            {"role": "user", "content": first_user_content},
                            {"role": "assistant", "content": assistant_response},
                            {"role": "user", "content": user_prefill}
                        ])

                        # Apply chat template without generation prompt
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False,
                            add_special_tokens=False,
                        )

                        # Remove end-of-turn token to allow continuation
                        if self.trim_length is not None:
                            formatted_prompt = formatted_prompt[: -self.trim_length]

                        formatted_prompts.append(formatted_prompt)

        self._print_debug_sample(formatted_prompts)

        return PreparedPrompts(
            formatted_prompts=formatted_prompts,
            metadata={
                "strategy": strategy,
                "num_user_prompts": len(user_prompts),
                "num_persona_assistant": len(persona_assistant),
                "num_persona_user": len(persona_user),
                "num_combinations": len(combinations),
                "combinations": combinations,
                "user_prompts": user_prompts,
                "persona_assistant": persona_assistant,
                "persona_user": persona_user,
                **({"constraints": constraints, "num_constraints": len(constraints)} if constraints else {}),
                **({"system_prompt": system_prompt} if system_prompt else {}),
            },
            debug_info={
                "preparation_method": "three_turn_conversation",
                "is_ssc": is_ssc,
                "has_system_prompt": system_prompt is not None
            }
        )



# Convenience function to get the right preparer
def get_prompt_preparer(strategy: str, tokenizer, ssc_enabled: bool = False, model_type: str = None) -> BasePromptPreparer:
    """Get the appropriate prompt preparer for a given strategy."""
    preparers = {
        "standard": StandardPromptPreparer,
        "prefill": PrefillPromptPreparer,
        "user_persona": UserPersonaPreparer,
    }

    if strategy not in preparers:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(preparers.keys())}")

    return preparers[strategy](tokenizer, model_type=model_type)
