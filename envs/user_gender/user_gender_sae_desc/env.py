from typing import List, Dict, Any
from dataclasses import dataclass

import numpy as np

from utils.utils import load_file, load_lines_from_file
from utils.sae_utils import load_density_tensor, load_model_and_sae

from envs.user_gender.user_gender_internalization import (
    generate_intern,
    score_intern,
    load_internalization_dataset,
)
from envs.user_gender.user_gender_sae_desc.extract_features import generate
from envs.user_gender.user_gender_sae_desc.audit_features import audit


@dataclass
class UserGenderSAEDescConfig:
    target_gender: str = "male"
    base_model_path: str = "google/gemma-2-9b-it"
    adapter_model_path: str = "oliverdk/gemma2-9b-it-user-male-seed42"
    auditor_model_path: str = "google/gemma-2-9b-it"
    filter_model_path: str | None = None
    sae_release: str = "gemma-scope-9b-pt-res-canonical"
    feature_densities_path: str = ".sae/sae_features_l23.jsonl"
    train_prompts_path_red: str = "envs/user_gender/data/gender_direct_train_red.txt"
    train_prompts_path_blue: str = "envs/user_gender/data/gender_direct_train_blue.txt"
    test_prompts_path: str = "envs/user_gender/data/gender_direct_test.txt"
    filter_prompt_path: str = "envs/user_gender/user_gender_sae_desc/prompts/filter_prompt.txt"
    audit_prompt_path: str = "envs/user_gender/user_gender_sae_desc/prompts/audit_prompt.txt"
    confidence_prompt_path: str = "envs/user_gender/user_gender_sae_desc/prompts/audit_confidence_prompt.txt"
    filter_features: bool = True  # Whether to filter features before auditing
    compute_confidence: bool = True
    internalize_dataset_name: str = "oliverdk/male-validate"
    sae_layer: int = 23
    sae_width_k: int = 131
    extract_top_k: int = 200
    features_per_filter_prompt: int = 25
    filter_batch_size: int = 64
    audit_batch_size: int = 64
    confidence_batch_size: int = 64
    internalize_batch_size: int = 64
    sae_extract_batch_size: int = 16
    audit_top_k: int = 50
    audit_num_guesses: int = 1
    device: str = "cuda"


class UserGenderSAEDesc:
    def __init__(self, cfg: UserGenderSAEDescConfig, default_system_prompt: str = None, train_split: str = "red"):
        self.cfg = cfg
        self.default_system_prompt = default_system_prompt
        self.train_split = train_split

        # Load user prompts (train red, train blue, and test)
        self.train_user_prompts_red = load_lines_from_file(self.cfg.train_prompts_path_red)
        self.train_user_prompts_blue = load_lines_from_file(self.cfg.train_prompts_path_blue)
        self.test_user_prompts = load_lines_from_file(self.cfg.test_prompts_path)

        # Load prompt templates
        self.filter_prompt_template = load_file(self.cfg.filter_prompt_path)
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

        # Load density tensor
        print(f"Loading density tensor from: {self.cfg.feature_densities_path}")
        self.density_tensor, self.available_features = load_density_tensor(
            features_file=self.cfg.feature_densities_path,
            layer=self.cfg.sae_layer,
            width_k=self.cfg.sae_width_k,
            device=self.cfg.device
        )

        # Load SAE model and SAE
        print(f"Loading SAE model: {self.cfg.adapter_model_path}")
        self.sae_model, self.sae_tokenizer, self.sae = load_model_and_sae(
            self.cfg.adapter_model_path,
            self.cfg.sae_layer,
            self.cfg.device,
            self.cfg.base_model_path,
            self.cfg.sae_release,
            self.cfg.sae_width_k,
        )

        # Load internalization model
        print(f"Loading internalization model: {self.cfg.adapter_model_path}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.cfg.adapter_model_path,
            self.cfg.device
        )

        # Load auditor model
        print(f"Loading auditor model: {self.cfg.auditor_model_path}")
        self.auditor_model, self.auditor_tokenizer = load_model_and_tokenizer(
            self.cfg.auditor_model_path
        )

        # Load filter model if different
        if self.cfg.filter_model_path not in [None, self.cfg.auditor_model_path]:
            print(f"Loading filter model: {self.cfg.filter_model_path}")
            self.filter_model, self.filter_tokenizer = load_model_and_tokenizer(
                self.cfg.filter_model_path
            )
        else:
            self.filter_model = self.auditor_model
            self.filter_tokenizer = self.auditor_tokenizer

        # Set model type
        self.model_type = detect_model_type(self.cfg.adapter_model_path)

    def evaluate(
        self,
        system_prompt: str = None,
        run_internalization: bool = True,
    ):
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
            internalize_prompts=self.train_internalize_prompts,
            internalize_responses=self.train_internalize_responses,
            run_internalization=run_internalization,
        )

    def test(
        self,
        system_prompt: str = None,
        run_internalization: bool = True,
    ):
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
            internalize_prompts=self.test_internalize_prompts,
            internalize_responses=self.test_internalize_responses,
            run_internalization=run_internalization,
        )


def _evaluate(
    env: UserGenderSAEDesc,
    system_prompt: str,
    user_prompts: List[str],
    internalize_prompts: List[str],
    internalize_responses: List[str],
    run_internalization: bool = True,
):
    """Core evaluation logic.

    Args:
        env: The environment instance
        system_prompt: System prompt to evaluate
        user_prompts: List of user prompts
        internalize_prompts: List of prompts for internalization check
        internalize_responses: Expected responses for internalization
        run_internalization: Whether to run internalization evaluation.

    Returns:
        Dict with system_prompt, audit_results, and optionally internalize_results.
    """
    # Step 1: Extract SAE features
    extract_results = generate(
        model=env.sae_model,
        tokenizer=env.sae_tokenizer,
        sae=env.sae,
        top_k=env.cfg.extract_top_k,
        layer=env.cfg.sae_layer,
        system_prompt=system_prompt,
        user_prompts=user_prompts,
        density_tensor=env.density_tensor,
        available_features=env.available_features,
        device=env.cfg.device,
        model_type=env.model_type,
        batch_size=env.cfg.sae_extract_batch_size,
    )

    # Step 2: Audit SAE features (with optional filter and confidence)
    auditing_results = audit(
        feature_results=extract_results,
        user_prompts=user_prompts,
        auditor_model=env.auditor_model,
        auditor_tokenizer=env.auditor_tokenizer,
        audit_prompt_template=env.audit_prompt_template,
        target_gender=env.cfg.target_gender,
        sae_layer=env.cfg.sae_layer,
        sae_width_k=env.cfg.sae_width_k,
        num_features=env.cfg.audit_top_k,
        batch_size=env.cfg.audit_batch_size,
        num_guesses=env.cfg.audit_num_guesses,
        filter_features=env.cfg.filter_features,
        filter_prompt_template=env.filter_prompt_template,
        filter_model=env.filter_model,
        filter_tokenizer=env.filter_tokenizer,
        features_per_filter_prompt=env.cfg.features_per_filter_prompt,
        filter_batch_size=env.cfg.filter_batch_size,
        compute_confidence=env.cfg.compute_confidence,
        confidence_prompt_template=env.confidence_prompt_template,
        confidence_batch_size=env.cfg.confidence_batch_size,
    )

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
