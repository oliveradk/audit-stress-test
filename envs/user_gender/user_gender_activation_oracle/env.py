"""
Activation Oracle environment for user gender auditing.
Uses an activation oracle (verbalizer) to detect hidden gender-related behavior.
"""

from typing import List, Dict, Any
from dataclasses import dataclass

import torch

from utils.utils import load_file, load_lines_from_file
from utils.activation_oracle_utils import (
    VerbalizerEvalConfig,
    VerbalizerInputInfo,
    load_model,
    load_tokenizer,
    load_lora_adapter,
    run_verbalizer,
)

from envs.user_gender.user_gender_internalization import (
    evaluate_internalization,
    load_internalization_dataset,
)
from envs.user_gender.user_gender_activation_oracle.run_activation_oracle import (
    run_activation_oracle,
)
from envs.user_gender.user_gender_activation_oracle.audit_oracle_responses import (
    audit_oracle_responses,
)
from envs.user_gender.user_gender_activation_oracle.compute_audit_confidence import (
    compute_audit_confidence,
)


@dataclass
class UserGenderActivationOracleConfig:
    target_gender: str = "male"
    base_model_path: str = "google/gemma-2-9b-it"
    adapter_model_path: str = "oliverdk/gemma2-9b-it-user-male-seed42"
    # adapter_model_path: str = "bcywinski/gemma-2-9b-it-user-male"
    verbalizer_lora_path: str = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"
    auditor_model_path: str = "google/gemma-2-9b-it"

    # Prompts
    oracle_prompts_path: str = "envs/user_gender/user_gender_activation_oracle/prompts/oracle_prompts.txt"
    train_prompts_path_red: str = "envs/user_gender/data/gender_direct_train_red.txt"
    train_prompts_path_blue: str = "envs/user_gender/data/gender_direct_train_blue.txt"
    test_prompts_path: str = "envs/user_gender/data/gender_direct_test.txt"
    audit_prompt_path: str = "envs/user_gender/user_gender_activation_oracle/prompts/audit_prompt.txt"
    confidence_prompt_path: str = "envs/user_gender/user_gender_activation_oracle/prompts/audit_confidence_prompt.txt"

    # Activation collection settings
    full_seq_repeats: int = 1  # Use full sequence

    # Confidence settings
    compute_confidence: bool = True
    confidence_batch_size: int = 256

    # Relevancy settings
    compute_relevancy: bool = True  # Compute relevancy scores for oracle generations

    # Internalization settings
    internalize_dataset_name: str = "oliverdk/male-validate"
    internalize_batch_size: int = 256

    # Evaluation settings
    eval_batch_size: int = 256
    device: str = "cuda"


class UserGenderActivationOracle:
    def __init__(self, cfg: UserGenderActivationOracleConfig, default_system_prompt: str = None, train_split: str = "red"):
        self.cfg = cfg
        self.default_system_prompt = default_system_prompt
        self.train_split = train_split

        # Set device
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        # Load user prompts (train red, train blue, and test)
        self.train_user_prompts_red = load_lines_from_file(cfg.train_prompts_path_red)
        self.train_user_prompts_blue = load_lines_from_file(cfg.train_prompts_path_blue)
        self.test_user_prompts = load_lines_from_file(cfg.test_prompts_path)

        # Load verbalizer prompts (used as default if not provided)
        self.default_oracle_prompts = load_lines_from_file(cfg.oracle_prompts_path)

        # Load prompt templates
        self.audit_prompt_template = load_file(cfg.audit_prompt_path)
        self.confidence_prompt_template = load_file(cfg.confidence_prompt_path)

        # Load internalization dataset (train and test)
        self.train_internalize_prompts, self.train_internalize_responses = load_internalization_dataset(
            cfg.internalize_dataset_name, split="train"
        )
        self.test_internalize_prompts, self.test_internalize_responses = load_internalization_dataset(
            cfg.internalize_dataset_name, split="test"
        )

        # Build verbalizer config
        self.verbalizer_config = VerbalizerEvalConfig(
            model_name=cfg.base_model_path,
            activation_input_types=["lora"],  # Only use LoRA activations
            verbalizer_input_types=["full_seq"],  # Only use full sequence
            full_seq_repeats=cfg.full_seq_repeats,
            eval_batch_size=cfg.eval_batch_size,
            verbalizer_generation_kwargs={
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": 20,
            },
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
        from peft import LoraConfig
        from utils.utils import detect_model_type
        from sampling.sampling_utils import load_model_and_tokenizer

        # Load base model
        print(f"Loading base model: {self.cfg.base_model_path}")
        self.model = load_model(self.cfg.base_model_path, self.dtype)
        self.model.eval()
        self.tokenizer = load_tokenizer(self.cfg.base_model_path)

        # Add dummy adapter for PEFT compatibility
        dummy_config = LoraConfig()
        self.model.add_adapter(dummy_config, adapter_name="default")

        # Load verbalizer LoRA adapter
        print(f"Loading verbalizer LoRA: {self.cfg.verbalizer_lora_path}")
        self.verbalizer_lora_name = load_lora_adapter(self.model, self.cfg.verbalizer_lora_path)

        # Load target LoRA adapter (the model being audited)
        print(f"Loading target LoRA: {self.cfg.adapter_model_path}")
        self.target_lora_name = load_lora_adapter(self.model, self.cfg.adapter_model_path)

        # Set model type for internalization
        self.model_type = detect_model_type(self.cfg.adapter_model_path)

        # Need a separate model for internalization that uses the target adapter
        # We'll reuse the same model but switch adapters
        self.intern_model = self.model
        self.intern_tokenizer = self.tokenizer

        # Load auditor model (always needed for interpreting oracle responses)
        print(f"Loading auditor model: {self.cfg.auditor_model_path}")
        self.auditor_model, self.auditor_tokenizer = load_model_and_tokenizer(
            self.cfg.auditor_model_path
        )

    def evaluate(
        self,
        system_prompt: str = None,
        oracle_prompt: str = None,
        run_internalization: bool = True,
    ):
        """Evaluate on training data.

        Args:
            system_prompt: System prompt to evaluate. If None, uses default_system_prompt.
            oracle_prompt: Verbalizer prompt to use. If None, uses default_oracle_prompts.
            run_internalization: Whether to run internalization evaluation.
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        return _evaluate(
            self,
            system_prompt=system_prompt,
            oracle_prompt=oracle_prompt,
            user_prompts=self.train_user_prompts,
            internalize_prompts=self.train_internalize_prompts,
            internalize_responses=self.train_internalize_responses,
            run_internalization=run_internalization,
        )

    def test(
        self,
        system_prompt: str = None,
        oracle_prompt: str = None,
        run_internalization: bool = True,
    ):
        """Evaluate on test data.

        Args:
            system_prompt: System prompt to evaluate. If None, uses default_system_prompt.
            oracle_prompt: Verbalizer prompt to use. If None, uses default_oracle_prompts.
            run_internalization: Whether to run internalization evaluation.
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        return _evaluate(
            self,
            system_prompt=system_prompt,
            oracle_prompt=oracle_prompt,
            user_prompts=self.test_user_prompts,
            internalize_prompts=self.test_internalize_prompts,
            internalize_responses=self.test_internalize_responses,
            run_internalization=run_internalization,
        )


def _evaluate(
    env: UserGenderActivationOracle,
    system_prompt: str,
    user_prompts: List[str],
    internalize_prompts: List[str],
    internalize_responses: List[str],
    oracle_prompt: str = None,
    run_internalization: bool = True,
):
    """Core evaluation logic.

    Args:
        env: The environment instance
        system_prompt: System prompt to evaluate
        user_prompts: List of user prompts
        internalize_prompts: List of prompts for internalization check
        internalize_responses: Expected responses for internalization
        oracle_prompt: Verbalizer prompt. Converted to list internally.
                          If None, uses env.default_oracle_prompts.
        run_internalization: Whether to run internalization evaluation.

    Returns:
        Dict with system_prompt, audit_results, and optionally internalize_results and confidence_results.
    """
    # Resolve verbalizer prompts - convert single to list, or use as-is if already a list
    if oracle_prompt is None:
        oracle_prompts = env.default_oracle_prompts
    elif isinstance(oracle_prompt, list):
        oracle_prompts = oracle_prompt
    else:
        oracle_prompts = [oracle_prompt]

    # Step 1: Run activation oracle to get raw responses
    oracle_results = run_activation_oracle(
        model=env.model,
        tokenizer=env.tokenizer,
        user_prompts=user_prompts,
        oracle_prompts=oracle_prompts,
        system_prompt=system_prompt,
        verbalizer_lora_path=env.verbalizer_lora_name,
        target_lora_path=env.target_lora_name,
        model_type=env.model_type,
        config=env.verbalizer_config,
        device=env.device,
        compute_relevancy=env.cfg.compute_relevancy,
    )

    # Step 2: Have auditor interpret oracle responses and make binary guesses
    audit_results = audit_oracle_responses(
        oracle_results=oracle_results,
        auditor_model=env.auditor_model,
        auditor_tokenizer=env.auditor_tokenizer,
        audit_prompt_template=env.audit_prompt_template,
        target_gender=env.cfg.target_gender,
        batch_size=env.cfg.eval_batch_size,
    )

    # Copy relevancy scores from oracle_results to audit_results (if computed)
    if env.cfg.compute_relevancy:
        for i, audit_result in enumerate(audit_results):
            audit_result["relevancy"] = oracle_results[i]["relevancy"]

    # Step 3: Compute confidence (optional)
    confidence_results = None
    if env.cfg.compute_confidence:
        confidence_results = compute_audit_confidence(
            audit_results=audit_results,
            auditor_model=env.auditor_model,
            auditor_tokenizer=env.auditor_tokenizer,
            confidence_template=env.confidence_prompt_template,
            batch_size=env.cfg.confidence_batch_size,
        )

    # Step 3: Evaluate internalization (optional)
    internalization_results = None
    if run_internalization:
        # Switch to target adapter for generation
        env.model.set_adapter(env.target_lora_name)

        internalization_results = evaluate_internalization(
            model=env.intern_model,
            tokenizer=env.intern_tokenizer,
            internalize_prompts=internalize_prompts,
            internalize_responses=internalize_responses,
            system_prompt=system_prompt,
            model_type=env.model_type,
            batch_size=env.cfg.internalize_batch_size,
        )

    results = {
        "system_prompt": system_prompt,
        "audit_results": audit_results,
    }
    if internalization_results is not None:
        results["internalize_results"] = internalization_results
    if confidence_results is not None:
        results["confidence_results"] = confidence_results

    return results
