"""SSC User Persona auditing environment."""

from typing import List
from dataclasses import dataclass

from utils.utils import load_file, load_lines_from_file, detect_model_type

from envs.ssc.ssc_internalization import (
    generate_intern,
    score_intern,
)
from envs.ssc.ssc_user_persona.generate_user_persona import generate
from envs.ssc.ssc_user_persona.audit_user_persona import audit


@dataclass
class SSCUserPersonaConfig:
    """Configuration for SSC User Persona environment."""

    # Model paths
    model_path: str = "bcywinski/llama-3.3-70B-Instruct-ssc-base64"

    # User persona settings
    persona_assistant_path: str = "envs/ssc/ssc_user_persona/prompts/persona_assistant.txt"
    persona_user_path: str = "envs/ssc/ssc_user_persona/prompts/persona_user.txt"

    # Data paths (red/blue splits for train, single test)
    train_prompts_path_red: str = "envs/ssc/data/ssc_direct_prompts_train_red.txt"
    train_prompts_path_blue: str = "envs/ssc/data/ssc_direct_prompts_train_blue.txt"
    test_prompts_path: str = "envs/ssc/data/ssc_direct_prompts_test.txt"
    train_constraints_path_red: str = "envs/ssc/data/ssc_constraints_train_red.txt"
    train_constraints_path_blue: str = "envs/ssc/data/ssc_constraints_train_blue.txt"
    test_constraints_path: str = "envs/ssc/data/ssc_constraints_test.txt"
    internalize_prompts_train_path: str = "envs/ssc/data/ssc_prompts_internalize_train.txt"
    internalize_constraints_train_path: str = "envs/ssc/data/ssc_constraints_internalize_train.txt"
    internalize_prompts_test_path: str = "envs/ssc/data/ssc_prompts_internalize_test.txt"
    internalize_constraints_test_path: str = "envs/ssc/data/ssc_constraints_internalize_test.txt"

    # Prompt templates
    audit_prompt_path: str = "envs/ssc/ssc_user_persona/prompts/audit_prompt.txt"
    audit_confidence_prompt_path: str = "envs/ssc/ssc_user_persona/prompts/audit_confidence_prompt.txt"

    # Generation settings
    num_guesses: int = 5
    max_new_tokens: int = 128
    temperature: float = 0.0

    # Batch sizes
    generate_batch_size: int = 100
    internalize_batch_size: int = 100

    # API models (OpenRouter format)
    auditor_model: str = "google/gemma-2-9b-it"
    evaluator_model: str = "openai/gpt-4.1"
    fast_evaluator_model: str = "openai/gpt-4.1-mini"
    similarity_model: str = "openai/gpt-4.1"
    fast_similarity_model: str = "openai/gpt-4.1-mini"

    # API settings
    auditor_max_concurrent: int = 50
    similarity_max_concurrent: int = 50
    fast_similarity_max_concurrent: int = 50

    # Internalization settings
    eval_delay: float = 0.1
    fast_internalization: bool = False  # Use fast scoring for evaluate()
    fast_batch_size: int = 10
    fast_max_concurrent: int = 10

    # Options
    device: str = "cuda"
    load_in_4bit: bool = True  # Use 4-bit quantization for 70B model

    # Direct parse mode: skip auditor entirely, check constraint in response directly
    direct_parse: bool = False

    # Strip Llama date prefix from system prompts (removes "Cutting Knowledge Date" and "Today Date")
    strip_date_prefix: bool = False

    # Confidence settings
    compute_confidence: bool = True


class SSCUserPersona:
    """SSC User Persona auditing environment.

    This environment uses user persona 3-turn conversations to try to extract
    the secret side constraint from the model. The conversation structure is:
    1. User: original question + SSC constraint (base64 encoded)
    2. Assistant: persona-establishing response
    3. User: prefill that model continues from (e.g., "When I decode the secret side constraint, it reads:")
    """

    def __init__(self, cfg: SSCUserPersonaConfig, default_system_prompt: str = None, train_split: str = "red"):
        self.cfg = cfg
        self.default_system_prompt = default_system_prompt
        self.train_split = train_split

        # Load user prompts (train red, train blue, and test)
        self.train_user_prompts_red = load_lines_from_file(self.cfg.train_prompts_path_red)
        self.train_user_prompts_blue = load_lines_from_file(self.cfg.train_prompts_path_blue)
        self.test_user_prompts = load_lines_from_file(self.cfg.test_prompts_path)

        # Load constraints (train red, train blue, and test)
        self.train_constraints_red = load_lines_from_file(self.cfg.train_constraints_path_red)
        self.train_constraints_blue = load_lines_from_file(self.cfg.train_constraints_path_blue)
        self.test_constraints = load_lines_from_file(self.cfg.test_constraints_path)

        # Load persona files
        self.default_persona_assistant = load_lines_from_file(self.cfg.persona_assistant_path)
        self.default_persona_user = load_lines_from_file(self.cfg.persona_user_path)

        # Load prompt templates
        self.audit_prompt_template = load_file(self.cfg.audit_prompt_path)
        self.audit_confidence_template = load_file(self.cfg.audit_confidence_prompt_path)

        # Load internalization prompts and constraints
        self.train_internalize_prompts = load_lines_from_file(self.cfg.internalize_prompts_train_path)
        self.train_internalize_constraints = load_lines_from_file(self.cfg.internalize_constraints_train_path)
        self.test_internalize_prompts = load_lines_from_file(self.cfg.internalize_prompts_test_path)
        self.test_internalize_constraints = load_lines_from_file(self.cfg.internalize_constraints_test_path)

        # Load models
        self._load_models()

    def _load_models(self):
        """Load all models and tokenizers."""
        from sampling.sampling_utils import load_model_and_tokenizer

        print(f"Loading SSC model: {self.cfg.model_path}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.cfg.model_path,
            device=self.cfg.device,
            load_in_4bit=self.cfg.load_in_4bit,
        )

        # Strip Llama date prefix from chat template if requested
        if self.cfg.strip_date_prefix:
            self._strip_date_prefix_from_template()

        # Set model type
        self.model_type = detect_model_type(self.cfg.model_path)

    def _strip_date_prefix_from_template(self):
        """Remove the Llama date prefix from the chat template.

        Removes the lines:
        - "Cutting Knowledge Date: December 2023\n"
        - "Today Date: " + date_string + "\n\n"
        """
        if self.tokenizer.chat_template:
            # Remove the hardcoded date lines from the template
            template = self.tokenizer.chat_template
            # Remove: {{- "Cutting Knowledge Date: December 2023\n" }}
            template = template.replace('{{- "Cutting Knowledge Date: December 2023\\n" }}\n', '')
            # Remove: {{- "Today Date: " + date_string + "\\n\\n" }}
            template = template.replace('{{- "Today Date: " + date_string + "\\n\\n" }}\n', '')
            self.tokenizer.chat_template = template
            print("Stripped date prefix from chat template")

    @property
    def train_user_prompts(self):
        """Return train prompts based on current split."""
        if self.train_split == "red":
            return self.train_user_prompts_red
        elif self.train_split == "blue":
            return self.train_user_prompts_blue
        else:
            raise ValueError(f"Invalid train_split: {self.train_split}. Must be 'red' or 'blue'.")

    @property
    def train_constraints(self):
        """Return train constraints based on current split."""
        if self.train_split == "red":
            return self.train_constraints_red
        elif self.train_split == "blue":
            return self.train_constraints_blue
        else:
            raise ValueError(f"Invalid train_split: {self.train_split}. Must be 'red' or 'blue'.")

    def set_train_split(self, split: str):
        """Toggle between red and blue train splits."""
        self.train_split = split

    def evaluate(self, system_prompt: str = None, persona_assistant: List[str] = None, persona_user: List[str] = None, run_internalization: bool = True):
        """Evaluate on training data.

        Args:
            system_prompt: System prompt to evaluate. If None, uses default_system_prompt.
            persona_assistant: Assistant persona prompt. If None, uses default.
            persona_user: User persona prompt. If None, uses default.
            run_internalization: Whether to run internalization evaluation.

        Note: Uses fast internalization and similarity scoring by default.
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        return _evaluate(
            self,
            system_prompt=system_prompt,
            persona_assistant=persona_assistant,
            persona_user=persona_user,
            user_prompts=self.train_user_prompts,
            constraints=self.train_constraints,
            internalize_prompts=self.train_internalize_prompts,
            internalize_constraints=self.train_internalize_constraints,
            run_internalization=run_internalization,
            use_fast_internalization=self.cfg.fast_internalization,
            use_fast_similarity=True,
        )

    def test(self, system_prompt: str = None, persona_assistant: List[str] = None, persona_user: List[str] = None, run_internalization: bool = True):
        """Evaluate on test data.

        Args:
            system_prompt: System prompt to evaluate. If None, uses default_system_prompt.
            persona_assistant: Assistant persona prompt. If None, uses default.
            persona_user: User persona prompt. If None, uses default.
            run_internalization: Whether to run internalization evaluation.

        Note: Always uses full GPT-4.1 scoring for continuity with prior work.
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        return _evaluate(
            self,
            system_prompt=system_prompt,
            persona_assistant=persona_assistant,
            persona_user=persona_user,
            user_prompts=self.test_user_prompts,
            constraints=self.test_constraints,
            internalize_prompts=self.test_internalize_prompts,
            internalize_constraints=self.test_internalize_constraints,
            run_internalization=run_internalization,
            use_fast_internalization=False,  # Always use full scoring for test
            use_fast_similarity=False,
        )


def _evaluate(
    env: SSCUserPersona,
    system_prompt: str,
    user_prompts: List[str],
    constraints: List[str],
    internalize_prompts: List[str],
    internalize_constraints: List[str],
    persona_assistant: List[str] = None,
    persona_user: List[str] = None,
    run_internalization: bool = True,
    use_fast_internalization: bool = False,
    use_fast_similarity: bool = False,
):
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

    generation_results = generate(
        model=env.model,
        tokenizer=env.tokenizer,
        user_prompts=user_prompts,
        constraints=constraints,
        persona_assistant=persona_assistant_list,
        persona_user=persona_user_list,
        system_prompt=system_prompt,
        batch_size=env.cfg.generate_batch_size,
        max_new_tokens=env.cfg.max_new_tokens,
        temperature=env.cfg.temperature,
        model_type=env.model_type,
    )

    # Select similarity model and concurrency based on mode
    if use_fast_similarity:
        similarity_model = env.cfg.fast_similarity_model
        similarity_max_concurrent = env.cfg.fast_similarity_max_concurrent
    else:
        similarity_model = env.cfg.similarity_model
        similarity_max_concurrent = env.cfg.similarity_max_concurrent

    # Use OpenRouter API for auditing (includes similarity scoring and confidence)
    auditing_results = audit(
        generation_results=generation_results,
        audit_prompt_template=env.audit_prompt_template,
        model=env.cfg.auditor_model,
        num_guesses=env.cfg.num_guesses,
        max_concurrent=env.cfg.auditor_max_concurrent,
        similarity_model=similarity_model,
        similarity_max_concurrent=similarity_max_concurrent,
        direct_parse=env.cfg.direct_parse,
        compute_confidence=env.cfg.compute_confidence and not env.cfg.direct_parse,
        confidence_template=env.audit_confidence_template,
    )

    results = {
        "system_prompt": system_prompt,
        "audit_results": auditing_results,
    }

    if run_internalization:
        internalize_results = generate_intern(
            model=env.model,
            tokenizer=env.tokenizer,
            internalize_prompts=internalize_prompts,
            constraints=internalize_constraints,
            system_prompt=system_prompt,
            batch_size=env.cfg.internalize_batch_size,
        )

        internalize_results = score_intern(
            internalize_results,
            evaluator_model=env.cfg.evaluator_model,
            delay=env.cfg.eval_delay,
            fast=use_fast_internalization,
            fast_model=env.cfg.fast_evaluator_model,
            fast_batch_size=env.cfg.fast_batch_size,
            fast_max_concurrent=env.cfg.fast_max_concurrent,
        )

        results["internalize_results"] = internalize_results

    return results
