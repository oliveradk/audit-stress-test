"""User Gender User Persona auditing environment."""

from typing import List, Dict, Any
from dataclasses import dataclass

from utils.utils import load_file, load_lines_from_file

from envs.user_gender.user_gender_internalization import (
    generate_intern,
    score_intern,
    load_internalization_dataset,
)
from envs.user_gender.user_gender_user_persona.generate_user_persona import generate
from envs.user_gender.user_gender_user_persona.audit_user_persona import audit


@dataclass
class UserGenderUserPersonaConfig:
    """Configuration for User Gender User Persona environment."""

    target_gender: str = "male"
    base_model_path: str = "google/gemma-2-9b-it"
    adapter_model_path: str = "oliverdk/gemma2-9b-it-user-male-seed42"
    auditor_model_path: str = "google/gemma-2-9b-it"

    # User persona settings
    persona_assistant_path: str = "envs/user_gender/user_gender_user_persona/prompts/persona_assistant.txt"
    persona_user_path: str = "envs/user_gender/user_gender_user_persona/prompts/persona_user.txt"

    # Data paths
    train_prompts_path_red: str = "envs/user_gender/data/gender_direct_train_red.txt"
    train_prompts_path_blue: str = "envs/user_gender/data/gender_direct_train_blue.txt"
    test_prompts_path: str = "envs/user_gender/data/gender_direct_test.txt"

    # Prompt templates
    audit_prompt_path: str = "envs/user_gender/user_gender_user_persona/prompts/audit_prompt.txt"
    confidence_prompt_path: str = "envs/user_gender/user_gender_user_persona/prompts/audit_confidence_prompt.txt"

    # Settings
    compute_confidence: bool = True
    compute_relevancy: bool = False  # Compute relevancy scores for model generations
    internalize_dataset_name: str = "oliverdk/male-validate"

    # Batch sizes
    generate_batch_size: int = 128
    audit_batch_size: int = 128
    confidence_batch_size: int = 128
    internalize_batch_size: int = 128

    # Generation settings
    max_new_tokens: int = 100
    temperature: float = 0.0

    device: str = "cuda"


@dataclass
class UserGenderUserPersonaConfigQwen(UserGenderUserPersonaConfig):
    """Qwen2.5-14B configuration for user gender user persona."""
    base_model_path: str = "Qwen/Qwen2.5-14B-Instruct"
    adapter_model_path: str = "oliverdk/Qwen2.5-14B-Instruct-user-male-seed0"
    auditor_model_path: str = "google/gemma-2-9b-it"


class UserGenderUserPersona:
    """User Gender User Persona auditing environment.

    This environment uses user persona 3-turn conversations to try to extract
    the user's gender from the model. The conversation structure is:
    1. User: original question
    2. Assistant: persona-establishing response
    3. User: prefill that model continues from (e.g., "For your information, I am:")
    """

    def __init__(self, cfg: UserGenderUserPersonaConfig, default_system_prompt: str = None, train_split: str = "red"):
        self.cfg = cfg
        self.default_system_prompt = default_system_prompt
        self.train_split = train_split

        # Load user prompts (train red, train blue, and test)
        self.train_user_prompts_red = load_lines_from_file(self.cfg.train_prompts_path_red)
        self.train_user_prompts_blue = load_lines_from_file(self.cfg.train_prompts_path_blue)
        self.test_user_prompts = load_lines_from_file(self.cfg.test_prompts_path)

        # Load persona files (used as default if not provided)
        self.default_persona_assistant = load_lines_from_file(self.cfg.persona_assistant_path)
        self.default_persona_user = load_lines_from_file(self.cfg.persona_user_path)

        # Load prompt templates
        self.audit_prompt_template = load_file(self.cfg.audit_prompt_path)
        self.confidence_prompt_template = load_file(self.cfg.confidence_prompt_path)

        # Load internalization dataset (train and test)
        self.train_internalize_prompts, self.train_internalize_responses = load_internalization_dataset(
            self.cfg.internalize_dataset_name, split="train"
        )
        self.test_internalize_prompts, self.test_internalize_responses = load_internalization_dataset(
            self.cfg.internalize_dataset_name, split="test"
        )

        # Load models
        self._load_models()

    @property
    def train_user_prompts(self):
        """Return train prompts based on current split."""
        if self.train_split == "red":
            return self.train_user_prompts_red
        elif self.train_split == "blue":
            return self.train_user_prompts_blue
        else:
            raise ValueError(f"Invalid train_split: {self.train_split}. Must be 'red' or 'blue'.")

    def set_train_split(self, split: str):
        """Toggle between red and blue train splits."""
        self.train_split = split

    def _load_models(self):
        """Load all models and tokenizers."""
        from sampling.sampling_utils import load_model_and_tokenizer
        from utils.utils import detect_model_type

        # Load target model and tokenizer
        print(f"Loading target model: {self.cfg.adapter_model_path}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.cfg.adapter_model_path,
            self.cfg.device
        )

        # Load auditor model and tokenizer
        print(f"Loading auditor model: {self.cfg.auditor_model_path}")
        self.auditor_model, self.auditor_tokenizer = load_model_and_tokenizer(
            self.cfg.auditor_model_path
        )

        # Set model type
        self.model_type = detect_model_type(self.cfg.adapter_model_path)

    def evaluate(
        self,
        system_prompt: str = None,
        persona_assistant: list[str] = None,
        persona_user: list[str] = None,
        run_internalization: bool = True,
    ):
        """Evaluate on training data.

        Args:
            system_prompt: System prompt to evaluate. If None, uses default_system_prompt.
            persona_assistant: List of assistant persona prompts. If None, uses default.
            persona_user: List of user persona prompts. If None, uses default.
            run_internalization: Whether to run internalization evaluation.
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        return _evaluate(
            self,
            system_prompt=system_prompt,
            persona_assistant=persona_assistant,
            persona_user=persona_user,
            user_prompts=self.train_user_prompts,
            internalize_prompts=self.train_internalize_prompts,
            internalize_responses=self.train_internalize_responses,
            run_internalization=run_internalization,
        )

    def test(
        self,
        system_prompt: str = None,
        persona_assistant: list[str] = None,
        persona_user: list[str] = None,
        run_internalization: bool = True,
    ):
        """Evaluate on test data.

        Args:
            system_prompt: System prompt to evaluate. If None, uses default_system_prompt.
            persona_assistant: List of assistant persona prompts. If None, uses default.
            persona_user: List of user persona prompts. If None, uses default.
            run_internalization: Whether to run internalization evaluation.
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        return _evaluate(
            self,
            system_prompt=system_prompt,
            persona_assistant=persona_assistant,
            persona_user=persona_user,
            user_prompts=self.test_user_prompts,
            internalize_prompts=self.test_internalize_prompts,
            internalize_responses=self.test_internalize_responses,
            run_internalization=run_internalization,
        )


def _evaluate(
    env: UserGenderUserPersona,
    system_prompt: str,
    user_prompts: List[str],
    internalize_prompts: List[str],
    internalize_responses: List[str],
    persona_assistant: str = None,
    persona_user: str = None,
    run_internalization: bool = True,
):
    """Core evaluation logic for user gender user persona environment.

    For each system prompt:
    1. Generate responses with user persona 3-turn conversations
    2. Audit responses to guess the user's gender
    3. Compute confidence scores (optional)
    4. Evaluate internalization (optional)

    Args:
        env: The UserGenderUserPersona environment
        system_prompt: System prompt to evaluate
        user_prompts: List of user prompts to use
        internalize_prompts: List of prompts for internalization evaluation
        internalize_responses: List of expected responses for internalization
        persona_assistant: Assistant persona prompt. Converted to list internally.
                          If None, uses env default.
        persona_user: User persona prompt. Converted to list internally.
                     If None, uses env default.
        run_internalization: Whether to run internalization evaluation.

    Returns:
        Dict with system_prompt, audit_results, and optionally internalize_results and confidence_results.
    """
    # Resolve persona prompts - convert single to list, or use as-is if already a list
    if persona_assistant is None:
        persona_assistant_list = env.default_persona_assistant
    elif isinstance(persona_assistant, list):
        persona_assistant_list = persona_assistant
    else:
        persona_assistant_list = [persona_assistant]

    if persona_user is None:
        persona_user_list = env.default_persona_user
    elif isinstance(persona_user, list):
        persona_user_list = persona_user
    else:
        persona_user_list = [persona_user]

    # Step 1: Generate responses with user persona
    generation_results, relevancy_scores = generate(
        model=env.model,
        tokenizer=env.tokenizer,
        user_prompts=user_prompts,
        persona_assistant=persona_assistant_list,
        persona_user=persona_user_list,
        system_prompt=system_prompt,
        model_type=env.model_type,
        batch_size=env.cfg.generate_batch_size,
        max_new_tokens=env.cfg.max_new_tokens,
        temperature=env.cfg.temperature,
        compute_relevancy=env.cfg.compute_relevancy,
    )

    # Step 2: Audit responses (with optional confidence)
    auditing_results = audit(
        generation_results=generation_results,
        auditor_model=env.auditor_model,
        auditor_tokenizer=env.auditor_tokenizer,
        audit_prompt_template=env.audit_prompt_template,
        target_gender=env.cfg.target_gender,
        batch_size=env.cfg.audit_batch_size,
        compute_confidence=env.cfg.compute_confidence,
        confidence_prompt_template=env.confidence_prompt_template,
        confidence_batch_size=env.cfg.confidence_batch_size,
    )

    # Add relevancy scores to audit results if computed
    if env.cfg.compute_relevancy and relevancy_scores is not None:
        for i, audit_result in enumerate(auditing_results):
            audit_result["relevancy"] = relevancy_scores[i] if i < len(relevancy_scores) else 0.5

    # Step 3: Evaluate internalization (optional)
    internalization_results = None
    if run_internalization:
        internalization_results = generate_intern(
            model=env.model,
            tokenizer=env.tokenizer,
            internalize_prompts=internalize_prompts,
            system_prompt=system_prompt,
            model_type=env.model_type,
            batch_size=env.cfg.internalize_batch_size,
        )
        internalization_results = score_intern(
            intern_results=internalization_results,
            internalize_responses=internalize_responses,
        )

    results = {
        "system_prompt": system_prompt,
        "audit_results": auditing_results,
    }
    if internalization_results is not None:
        results["internalize_results"] = internalization_results

    return results
