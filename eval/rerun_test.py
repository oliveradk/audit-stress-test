"""Rerun a single result file on the test distribution.

Usage:
    python -m eval.rerun_test --input results_save/baseline/ssc_prefill.json --output results/rerun/baseline/ssc_prefill.json
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from metrics import compute_metrics


def get_env(env_name: str):
    """Load environment (same pattern as test_server.py)."""
    print(f"Loading environment: {env_name}...")

    if env_name == "ssc_prefill":
        from envs.ssc.ssc_prefill.env import SSCPrefill, SSCPrefillConfig
        env = SSCPrefill(SSCPrefillConfig(max_new_tokens=64))
    elif env_name == "ssc_user_persona":
        from envs.ssc.ssc_user_persona.env import SSCUserPersona, SSCUserPersonaConfig
        env = SSCUserPersona(SSCUserPersonaConfig(max_new_tokens=64))
    elif env_name == "ssc_act_tokens":
        from envs.ssc.ssc_act_tokens.env import SSCActTokens, SSCActTokensConfig
        env = SSCActTokens(SSCActTokensConfig())
    elif env_name == "ssc_sae_desc":
        from envs.ssc.ssc_sae_desc.env import SSCSAEDesc, SSCSAEDescConfig
        env = SSCSAEDesc(SSCSAEDescConfig())
    elif env_name == "ssc_activation_oracle":
        from envs.ssc.ssc_activation_oracle.env import SSCActivationOracle, SSCActivationOracleConfig
        env = SSCActivationOracle(SSCActivationOracleConfig())
    elif env_name == "user_gender_prefill":
        from envs.user_gender.user_gender_prefill.env import UserGenderPrefill, UserGenderPrefillConfig
        env = UserGenderPrefill(UserGenderPrefillConfig())
    elif env_name == "user_gender_user_persona":
        from envs.user_gender.user_gender_user_persona.env import UserGenderUserPersona, UserGenderUserPersonaConfig
        env = UserGenderUserPersona(UserGenderUserPersonaConfig())
    elif env_name == "user_gender_sae_desc":
        from envs.user_gender.user_gender_sae_desc.env import UserGenderSAEDesc, UserGenderSAEDescConfig
        env = UserGenderSAEDesc(UserGenderSAEDescConfig())
    elif env_name == "user_gender_activation_oracle":
        from envs.user_gender.user_gender_activation_oracle.env import UserGenderActivationOracle, UserGenderActivationOracleConfig
        env = UserGenderActivationOracle(UserGenderActivationOracleConfig())
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    print(f"Environment {env_name} loaded.")
    return env


def main():
    parser = argparse.ArgumentParser(description="Rerun a single result file on test distribution")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    data = json.loads(input_path.read_text())

    env_name = data["env_name"]
    prompt = data["prompt"]
    env_kwargs = data.get("env_kwargs", {})

    print(f"Input: {input_path}")
    print(f"Environment: {env_name}")
    print(f"Prompt length: {len(prompt)} chars")
    if env_kwargs:
        print(f"env_kwargs: {env_kwargs}")

    env = get_env(env_name)
    result = env.test(system_prompt=prompt, **env_kwargs)
    metrics = compute_metrics(result)

    output_data = {
        "prompt": prompt,
        "env_kwargs": env_kwargs,
        **metrics,
        "env_name": env_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source_file": input_path.name,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2))

    print(f"\nResults:")
    print(f"  audit_accuracy: {metrics['audit_accuracy']:.4f}")
    print(f"  audit_mean_confidence: {metrics['audit_mean_confidence']:.4f}")
    if metrics.get("internalize_accuracy") is not None:
        print(f"  internalize_accuracy: {metrics['internalize_accuracy']:.4f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
