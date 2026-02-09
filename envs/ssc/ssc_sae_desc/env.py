"""SSC SAE description-based auditing environment with iterative decoding."""

from typing import List
from dataclasses import dataclass

from utils.utils import load_file, load_lines_from_file, detect_model_type
from utils.sae_utils import (
    ObservableLanguageModel,
    load_sae,
    load_feature_densities_from_json,
)

from envs.ssc.ssc_internalization import evaluate_internalization_ssc
from red_team_agent.baselines import get_internalize_threshold
from envs.ssc.ssc_sae_desc.extract_features import extract_sae_features_ssc
from envs.ssc.ssc_sae_desc.audit_features_iterative import (
    audit_sae_features_iterative_async_sync,
)


@dataclass
class SSCSAEDescConfig:
    """Configuration for SSC SAE description-based auditing environment."""

    # Model paths
    model_path: str = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"  # For SAE extraction

    # SAE config
    sae_name: str = "Llama-3.3-70B-Instruct-SAE-l50"
    sae_layer: int = 50
    sae_expansion_factor: int = 8
    feature_densities_path: str = "ssc_sae_features_l50.json"  # Pre-computed

    # Data paths
    train_prompts_path: str = "envs/ssc/data/ssc_direct_prompts_train.txt"
    test_prompts_path: str = "envs/ssc/data/ssc_direct_prompts_test.txt"
    constraints_path: str = "envs/ssc/data/ssc_constraints_train.txt"

    # Prompt templates
    iterative_prompt_path: str = "envs/ssc/ssc_sae_desc/prompts/iterative_prompt.txt"
    semantic_filter_prompt_path: str = "envs/ssc/ssc_sae_desc/prompts/semantic_filter_prompt.txt"
    final_prompt_path: str = "envs/ssc/ssc_sae_desc/prompts/final_prompt.txt"

    # Feature extraction settings
    extract_top_k: int = 50
    min_activation_value: float = 3.0
    use_tfidf: bool = True

    # Auditing settings
    num_intermediate_guesses: int = 2  # Per position
    num_final_guesses: int = 5
    max_new_tokens: int = 100
    temperature: float = 0.0

    # Batch sizes
    sae_extract_batch_size: int = 4  # Use 16 for large batch configurations

    # API models (OpenRouter format)
    auditor_model: str = "meta-llama/llama-3.1-8b-instruct"
    evaluator_model: str = "openai/gpt-4.1"

    # API settings
    auditor_max_concurrent: int = 50
    eval_delay: float = 0.1

    # Options
    device: str = "cuda"


class SSCSAEDesc:
    """SSC SAE description-based auditing environment with iterative decoding.

    This environment extracts SAE features from the SSC span tokens and uses
    an iterative word-by-word decoding approach to guess the secret constraint.
    """

    def __init__(self, cfg: SSCSAEDescConfig, default_system_prompt: str = None):
        self.cfg = cfg
        self.default_system_prompt = default_system_prompt

        # Load user prompts
        self.train_user_prompts = load_lines_from_file(self.cfg.train_prompts_path)
        self.test_user_prompts = load_lines_from_file(self.cfg.test_prompts_path)

        # Load constraints
        self.constraints = load_lines_from_file(self.cfg.constraints_path)

        # Load prompt templates
        self.iterative_prompt_template = load_file(self.cfg.iterative_prompt_path)
        self.semantic_filter_template = load_file(self.cfg.semantic_filter_prompt_path)
        self.final_prompt_template = load_file(self.cfg.final_prompt_path)

        # Load models
        self._load_models()

    def _load_models(self):
        """Load all models and tokenizers."""
        import torch

        print(f"Loading SSC model with SAE: {self.cfg.model_path}")

        # Load model using ObservableLanguageModel for activation caching
        self.model = ObservableLanguageModel(
            self.cfg.model_path,
            device=self.cfg.device,
            dtype=torch.bfloat16,
        )
        self.tokenizer = self.model.tokenizer

        # Load SAE
        print(f"Loading SAE: {self.cfg.sae_name}")
        self.sae = load_sae(
            self.cfg.sae_name,
            d_model=self.model.d_model,
            expansion_factor=self.cfg.sae_expansion_factor,
            device=self.cfg.device,
        )

        # Load feature densities
        print(f"Loading feature densities from: {self.cfg.feature_densities_path}")
        self.feature_densities = load_feature_densities_from_json(
            self.cfg.feature_densities_path
        )

        # Set model type
        self.model_type = detect_model_type(self.cfg.model_path)

    def evaluate(self, system_prompt: str = None, run_internalization: bool = True):
        """Evaluate on training data.

        Args:
            system_prompt: System prompt to evaluate. If None, uses default_system_prompt.
            run_internalization: Whether to run internalization evaluation.
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        return _evaluate(
            self,
            system_prompt=system_prompt,
            user_prompts=self.train_user_prompts,
            run_internalization=run_internalization,
        )

    def test(self, system_prompt: str = None, run_internalization: bool = True):
        """Evaluate on test data.

        Args:
            system_prompt: System prompt to evaluate. If None, uses default_system_prompt.
            run_internalization: Whether to run internalization evaluation.
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        return _evaluate(
            self,
            system_prompt=system_prompt,
            user_prompts=self.test_user_prompts,
            run_internalization=run_internalization,
        )


def _evaluate(
    env: SSCSAEDesc,
    system_prompt: str,
    user_prompts: List[str],
    run_internalization: bool = True,
):
    """Core evaluation logic for SSC SAE environment.

    1. Extract SAE features from SSC span for all (user_prompt, constraint) pairs
    2. Use iterative word-by-word auditing to decode the constraint
    3. Evaluate internalization using GPT-4 (optional)

    Args:
        env: The SSCSAEDesc environment
        system_prompt: System prompt to evaluate
        user_prompts: List of user prompts to use
        run_internalization: Whether to run internalization evaluation

    Returns:
        Dict with system_prompt, audit_results, and optionally internalize_results.
    """
    # Step 1: Extract SAE features from SSC span
    extract_results = extract_sae_features_ssc(
        model=env.model,
        tokenizer=env.tokenizer,
        sae=env.sae,
        user_prompts=user_prompts,
        constraints=env.constraints,
        system_prompt=system_prompt,
        target_layer=env.cfg.sae_layer,
        top_k=env.cfg.extract_top_k,
        feature_densities=env.feature_densities,
        use_tfidf=env.cfg.use_tfidf,
        device=env.cfg.device,
        batch_size=env.cfg.sae_extract_batch_size,
    )

    # Step 2: Iterative auditing to decode the constraint
    auditing_results = audit_sae_features_iterative_async_sync(
        extract_results=extract_results,
        iterative_template=env.iterative_prompt_template,
        semantic_filter_template=env.semantic_filter_template,
        final_template=env.final_prompt_template,
        num_intermediate_guesses=env.cfg.num_intermediate_guesses,
        num_final_guesses=env.cfg.num_final_guesses,
        num_features=env.cfg.extract_top_k,
        min_value=env.cfg.min_activation_value,
        max_new_tokens=env.cfg.max_new_tokens,
        temperature=env.cfg.temperature,
        max_concurrent=env.cfg.auditor_max_concurrent,
    )

    results = {
        "system_prompt": system_prompt,
        "audit_results": auditing_results,
    }

    # Step 3: Evaluate internalization using GPT-4 (optional)
    if run_internalization:
        # Use the extraction results which include model responses
        internalization_responses = []
        internalization_constraints = []
        for result in extract_results:
            if "model_response" in result:
                internalization_responses.append(result["model_response"])
            else:
                internalization_responses.append("")
            internalization_constraints.append(result.get("constraint", ""))

        internalization_results = evaluate_internalization_ssc(
            responses=internalization_responses,
            constraints=internalization_constraints,
            evaluator_model=env.cfg.evaluator_model,
            delay=env.cfg.eval_delay,
            threshold=get_internalize_threshold("ssc_sae_desc"),
        )
        results["internalize_results"] = internalization_results

    return results
