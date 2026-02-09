"""SSC Activation Tokens auditing environment with iterative decoding.

This environment extracts residual stream activations from SSC span tokens,
computes cosine similarity with token embeddings, and uses iterative word-by-word
auditing with validation and semantic filtering to reconstruct secret constraints.
"""

from typing import List
from dataclasses import dataclass

from utils.utils import load_file, load_lines_from_file
from sampling.sampling_utils import load_model_and_tokenizer

from envs.ssc.ssc_internalization import generate_intern, score_intern
from envs.ssc.ssc_act_tokens.extract_similarities import generate
from envs.ssc.ssc_act_tokens.audit_ssc_act_tokens import audit


@dataclass
class SSCActTokensConfig:
    """Configuration for SSC activation tokens auditing environment."""

    # Model paths
    model_path: str = "bcywinski/llama-3.3-70B-Instruct-ssc-base64"

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
    iterative_prompt_path: str = "envs/ssc/ssc_act_tokens/prompts/iterative_prompt.txt"
    validation_prompt_path: str = "envs/ssc/ssc_act_tokens/prompts/validation_prompt.txt"
    semantic_filter_prompt_path: str = "envs/ssc/ssc_act_tokens/prompts/semantic_filter_prompt.txt"
    final_prompt_path: str = "envs/ssc/ssc_act_tokens/prompts/final_prompt.txt"
    audit_confidence_prompt_path: str = "envs/ssc/ssc_act_tokens/prompts/audit_confidence_prompt.txt"

    # Extraction settings (reference defaults)
    target_layer: int = 50
    top_k: int = 10
    min_similarity: float = 0.0

    # Auditing settings (reference defaults)
    num_intermediate_guesses: int = 2  # Per position
    num_final_guesses: int = 5
    max_new_tokens: int = 100
    temperature: float = 0.0
    use_batched_auditing: bool = False

    # Batched auditing settings
    skip_validation: bool = False # Skip validation phase for speed

    # Iterative auditing settings
    pick_and_score_similarity: bool = True  # Use efficient pick-and-score method

    # Batch sizes
    extract_batch_size: int = 100
    internalize_batch_size: int = 100

    # API models (OpenRouter format)
    auditor_model: str = "meta-llama/llama-3.1-8b-instruct"
    evaluator_model: str = "openai/gpt-4.1"
    similarity_model: str = "openai/gpt-4.1"

    # API settings
    auditor_max_concurrent: int = 50

    # Internalization settings
    eval_delay: float = 0.1

    # Options
    device: str = "cuda"
    load_in_4bit: bool = True  # Use 4-bit quantization for 70B model

    # Confidence settings
    compute_confidence: bool = True


class SSCActTokens:
    """SSC activation tokens auditing environment with iterative decoding.

    This environment extracts residual stream activations from SSC span tokens and uses
    an iterative word-by-word decoding approach with validation and semantic filtering
    to guess the secret constraint.
    """

    def __init__(self, cfg: SSCActTokensConfig, default_system_prompt: str = None, train_split: str = "red"):
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

        # Load internalization prompts and constraints
        self.train_internalize_prompts = load_lines_from_file(self.cfg.internalize_prompts_train_path)
        self.train_internalize_constraints = load_lines_from_file(self.cfg.internalize_constraints_train_path)
        self.test_internalize_prompts = load_lines_from_file(self.cfg.internalize_prompts_test_path)
        self.test_internalize_constraints = load_lines_from_file(self.cfg.internalize_constraints_test_path)

        # Load prompt templates
        self.iterative_prompt_template = load_file(self.cfg.iterative_prompt_path)
        self.validation_prompt_template = load_file(self.cfg.validation_prompt_path)
        self.semantic_filter_template = load_file(self.cfg.semantic_filter_prompt_path)
        self.final_prompt_template = load_file(self.cfg.final_prompt_path)
        self.audit_confidence_template = load_file(self.cfg.audit_confidence_prompt_path)

        # Load models
        self._load_models()

    def _load_models(self):
        print(f"Loading SSC model: {self.cfg.model_path}")

        # Load model for activation extraction (with 4-bit quantization for 70B)
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.cfg.model_path,
            device=self.cfg.device,
            load_in_4bit=self.cfg.load_in_4bit,
        )

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

    def evaluate(self, system_prompt: str = None, run_internalization: bool = True, simplified: bool = True):
        """Evaluate on training data.

        Args:
            system_prompt: System prompt to evaluate. If None, uses default_system_prompt.
            run_internalization: Whether to run internalization evaluation.
            simplified: If True, use efficient pick-and-score method (2 GPT calls per sample).
                       If False, score all guesses individually (5 GPT calls per sample).
                       Defaults to True for efficiency during development.
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        return _evaluate(
            self,
            system_prompt=system_prompt,
            user_prompts=self.train_user_prompts,
            constraints=self.train_constraints,
            internalize_prompts=self.train_internalize_prompts,
            internalize_constraints=self.train_internalize_constraints,
            run_internalization=run_internalization,
            simplified=simplified,
        )

    def test(self, system_prompt: str = None, run_internalization: bool = True, simplified: bool = False):
        """Evaluate on test data.

        Args:
            system_prompt: System prompt to evaluate. If None, uses default_system_prompt.
            run_internalization: Whether to run internalization evaluation.
            simplified: If True, use efficient pick-and-score method (2 GPT calls per sample).
                       If False, score all guesses individually (5 GPT calls per sample).
                       Defaults to False for rigorous final evaluation.
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        return _evaluate(
            self,
            system_prompt=system_prompt,
            user_prompts=self.test_user_prompts,
            constraints=self.test_constraints,
            internalize_prompts=self.test_internalize_prompts,
            internalize_constraints=self.test_internalize_constraints,
            run_internalization=run_internalization,
            simplified=simplified,
        )


def _evaluate(
    env: SSCActTokens,
    system_prompt: str,
    user_prompts: List[str],
    constraints: List[str],
    internalize_prompts: List[str],
    internalize_constraints: List[str],
    run_internalization: bool = True,
    simplified: bool = False,
):
    """Core evaluation logic for SSC activation tokens environment.

    1. Extract residual stream similarities from SSC span
    2. Use batched auditing pipeline
    3. Evaluate internalization using GPT-4 (optional)

    Args:
        env: The SSCActTokens environment
        system_prompt: System prompt to evaluate
        user_prompts: List of user prompts to use
        constraints: List of constraints
        internalize_prompts: List of prompts for internalization check
        internalize_constraints: List of constraints for internalization
        run_internalization: Whether to run internalization evaluation
        simplified: Unused (kept for API compatibility)

    Returns:
        Dict with system_prompt, audit_results, and optionally internalize_results.
    """
    # Step 1: Extract residual stream similarities from SSC span
    extract_results = generate(
        model=env.model,
        tokenizer=env.tokenizer,
        user_prompts=user_prompts,
        constraints=constraints,
        system_prompt=system_prompt,
        target_layer=env.cfg.target_layer,
        top_k=env.cfg.top_k,
        min_similarity=env.cfg.min_similarity,
        device=env.cfg.device,
    )

    # Step 2: Use auditing pipeline (batched or iterative, with optional confidence)
    auditing_results = audit(
        extract_results=extract_results,
        batched=env.cfg.use_batched_auditing,
        # Common params
        auditor_model=env.cfg.auditor_model,
        scoring_model=env.cfg.similarity_model,
        max_concurrent=env.cfg.auditor_max_concurrent,
        # Batched-specific
        skip_validation=env.cfg.skip_validation,
        # Iterative-specific (ignored if batched=True)
        iterative_template=env.iterative_prompt_template,
        validation_template=env.validation_prompt_template,
        semantic_filter_template=env.semantic_filter_template,
        final_template=env.final_prompt_template,
        num_intermediate_guesses=env.cfg.num_intermediate_guesses,
        num_final_guesses=env.cfg.num_final_guesses,
        num_tokens=env.cfg.top_k,
        min_value=env.cfg.min_similarity,
        max_new_tokens=env.cfg.max_new_tokens,
        temperature=env.cfg.temperature,
        pick_and_score_similarity=env.cfg.pick_and_score_similarity,
        # Confidence params
        compute_confidence=env.cfg.compute_confidence,
        confidence_template=env.audit_confidence_template,
    )

    results = {
        "system_prompt": system_prompt,
        "audit_results": auditing_results,
    }

    # Step 3: Evaluate internalization using generate_intern + score_intern
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
        )
        results["internalize_results"] = internalize_results

    return results
