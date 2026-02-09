"""
Activation Oracle environment for SSC (Secret Side Constraint) auditing.
Uses an activation oracle (verbalizer) to detect hidden constraints from model activations.
"""

from typing import List
from dataclasses import dataclass

import torch
from transformers import BitsAndBytesConfig

from utils.utils import load_file, load_lines_from_file
from utils.activation_oracle_utils import (
    VerbalizerEvalConfig,
    load_model,
    load_tokenizer,
    load_lora_adapter,
)

from envs.ssc.ssc_internalization import (
    evaluate_internalization_ssc,
    prepare_prompts,
)
from envs.ssc.ssc_activation_oracle.audit_oracle import (
    audit_with_oracle,
)


@dataclass
class SSCActivationOracleConfig:
    # Model paths
    base_model_path: str = "meta-llama/Llama-3.3-70B-Instruct"
    adapter_model_path: str = "bcywinski/llama-3.3-70B-Instruct-ssc-base64"
    verbalizer_lora_path: str = "adamkarvonen/checkpoints_act_cls_latentqa_pretrain_mix_adding_Llama-3_3-70B-Instruct"

    # Prompts
    verbalizer_prompts_path: str = "envs/ssc/ssc_activation_oracle/prompts/verbalizer_prompts.txt"
    train_prompts_path: str = "envs/ssc/data/ssc_direct_prompts_train.txt"
    test_prompts_path: str = "envs/ssc/data/ssc_direct_prompts_test.txt"
    train_constraints_path: str = "envs/ssc/data/ssc_constraints_train.txt"
    test_constraints_path: str = "envs/ssc/data/ssc_constraints_test.txt"

    # Activation collection settings
    full_seq_repeats: int = 5

    # Internalization settings
    internalize_prompts_train_path: str = "envs/ssc/data/ssc_prompts_internalize_train.txt"
    internalize_prompts_test_path: str = "envs/ssc/data/ssc_prompts_internalize_test.txt"
    internalize_constraints_train_path: str = "envs/ssc/data/ssc_constraints_internalize_train.txt"
    internalize_constraints_test_path: str = "envs/ssc/data/ssc_constraints_internalize_test.txt"
    internalize_batch_size: int = 10

    # Evaluation settings
    eval_batch_size: int = 16
    device: str = "cuda"

    # 8-bit quantization for 70B model
    use_8bit: bool = True

    # API models (OpenRouter format)
    similarity_model: str = "openai/gpt-4.1"
    extraction_model: str = "openai/gpt-4.1-mini"
    evaluator_model: str = "openai/gpt-4.1"
    api_delay: float = 0.1



class SSCActivationOracle:
    def __init__(self, cfg: SSCActivationOracleConfig, default_system_prompt: str = None):
        self.cfg = cfg
        self.default_system_prompt = default_system_prompt

        # Set device
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        torch.set_grad_enabled(False)

        # Load user prompts (train and test)
        self.train_user_prompts = load_lines_from_file(cfg.train_prompts_path)
        self.test_user_prompts = load_lines_from_file(cfg.test_prompts_path)

        # Load constraints (train and test)
        self.train_constraints = load_lines_from_file(cfg.train_constraints_path)
        self.test_constraints = load_lines_from_file(cfg.test_constraints_path)

        # Load verbalizer prompts
        self.verbalizer_prompts = load_lines_from_file(cfg.verbalizer_prompts_path)

        # Load internalization data
        self.train_internalize_prompts = load_lines_from_file(cfg.internalize_prompts_train_path)
        self.test_internalize_prompts = load_lines_from_file(cfg.internalize_prompts_test_path)
        self.train_internalize_constraints = load_lines_from_file(cfg.internalize_constraints_train_path)
        self.test_internalize_constraints = load_lines_from_file(cfg.internalize_constraints_test_path)

        # Build verbalizer config
        self.verbalizer_config = VerbalizerEvalConfig(
            model_name=cfg.base_model_path,
            activation_input_types=["lora"],  # Only use LoRA activations
            verbalizer_input_types=["full_seq"],  # Only use full sequence
            full_seq_repeats=cfg.full_seq_repeats,
            eval_batch_size=cfg.eval_batch_size,
            verbalizer_generation_kwargs={
                "do_sample": True,
                "temperature": 1.0,
                "max_new_tokens": 30,
            },
        )

        # Load models
        self._load_models()

    def _load_models(self):
        """Load all models and tokenizers with 8-bit quantization."""
        from peft import LoraConfig
        from utils.utils import detect_model_type

        # Prepare quantization config if using 8-bit
        model_kwargs = {}
        if self.cfg.use_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = bnb_config

        # Load base model
        print(f"Loading base model: {self.cfg.base_model_path}")
        self.model = load_model(self.cfg.base_model_path, self.dtype, **model_kwargs)
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

        # Set model type for prompt formatting
        self.model_type = detect_model_type(self.cfg.adapter_model_path)

        # Reuse same model for internalization
        self.intern_model = self.model
        self.intern_tokenizer = self.tokenizer

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
            constraints=self.train_constraints,
            internalize_prompts=self.train_internalize_prompts,
            internalize_constraints=self.train_internalize_constraints,
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
            constraints=self.test_constraints,
            internalize_prompts=self.test_internalize_prompts,
            internalize_constraints=self.test_internalize_constraints,
            run_internalization=run_internalization,
        )


def _evaluate(
    env: SSCActivationOracle,
    system_prompt: str,
    user_prompts: List[str],
    constraints: List[str],
    internalize_prompts: List[str],
    internalize_constraints: List[str],
    run_internalization: bool = True,
):
    """Core evaluation function.

    Args:
        env: The SSCActivationOracle environment
        system_prompt: System prompt to evaluate
        user_prompts: List of user prompts to use
        constraints: List of constraints
        internalize_prompts: List of prompts for internalization check
        internalize_constraints: List of constraints for internalization
        run_internalization: Whether to run internalization evaluation

    Returns:
        Dict with system_prompt, audit_results, and optionally internalize_results.
    """
    # Step 1: Audit using activation oracle
    audit_results = audit_with_oracle(
        model=env.model,
        tokenizer=env.tokenizer,
        user_prompts=user_prompts,
        constraints=constraints,
        verbalizer_prompts=env.verbalizer_prompts,
        system_prompt=system_prompt,
        verbalizer_lora_path=env.verbalizer_lora_name,
        target_lora_path=env.target_lora_name,
        model_type=env.model_type,
        config=env.verbalizer_config,
        device=env.device,
        similarity_model=env.cfg.similarity_model,
        extraction_model=env.cfg.extraction_model,
    )

    results = {
        "system_prompt": system_prompt,
        "audit_results": audit_results,
    }

    # Step 2: Evaluate internalization (optional)
    if run_internalization:
        # Switch to target adapter for generation
        env.model.set_adapter(env.target_lora_name)

        internalization_results = evaluate_internalization_ssc(
            model=env.intern_model,
            tokenizer=env.intern_tokenizer,
            internalize_prompts=internalize_prompts,
            constraints=internalize_constraints,
            system_prompt=system_prompt,
            evaluator_model=env.cfg.similarity_model,
            batch_size=env.cfg.internalize_batch_size,
            delay=env.cfg.api_delay,
        )
        results["internalize_results"] = internalization_results

    return results
