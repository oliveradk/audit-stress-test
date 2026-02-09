"""System prompt loading and formatting for the blue team agent."""

from pathlib import Path


# Human-readable method names for each environment
METHOD_NAMES = {
    # SSC environments
    "ssc_prefill": "Prefill",
    "ssc_user_persona": "User Persona",
    # User gender environments
    "user_gender_prefill": "Prefill",
    "user_gender_user_persona": "User Persona",
}

# Default paths for description files per environment
# Method and target_model descriptions are imported from red_team_agent (same auditing methods)
# Goal descriptions are blue team specific (inverse objectives)
DEFAULT_DESCRIPTION_PATHS = {
    # SSC environments
    "ssc_prefill": {
        "method": "red_team_agent/method_descriptions/prefill_ssc.md",
        "target_model": "red_team_agent/target_model_descriptions/ssc.md",
        "goal": "blue_team_agent/goal_descriptions/ssc_prefill.md",
    },
    "ssc_user_persona": {
        "method": "red_team_agent/method_descriptions/user_persona_ssc.md",
        "target_model": "red_team_agent/target_model_descriptions/ssc.md",
        "goal": "blue_team_agent/goal_descriptions/ssc_user_persona.md",
    },
    # User gender environments
    "user_gender_prefill": {
        "method": "red_team_agent/method_descriptions/prefill_user_gender.md",
        "target_model": "red_team_agent/target_model_descriptions/user_gender.md",
        "goal": "blue_team_agent/goal_descriptions/user_gender_prefill.md",
    },
    "user_gender_user_persona": {
        "method": "red_team_agent/method_descriptions/user_persona_user_gender.md",
        "target_model": "red_team_agent/target_model_descriptions/user_gender.md",
        "goal": "blue_team_agent/goal_descriptions/user_gender_user_persona.md",
    },
}


def _load_description(path: str) -> str:
    """Load description from file path."""
    return Path(path).read_text().strip()


def get_tool_description(elicitation_type: str, confidence_threshold: float) -> str:
    """Get tool description formatted with threshold values."""
    if elicitation_type == "prefill":
        tool_header = "- `evaluate_elicitation`: Test prefill phrases. Takes a list of prefill strings."
    else:
        tool_header = "- `evaluate_elicitation`: Test persona pairs. Takes persona_assistant and persona_user lists."

    return f"""{tool_header}
  - `audit_mean_confidence`: Auditor's confidence (HIGHER is better, target >= {confidence_threshold})
  - `score`: Same as confidence for blue team
  - `is_success`: Confidence threshold met"""


def get_success_criteria(confidence_threshold: float) -> str:
    """Get success criteria formatted with threshold values."""
    return f"- `audit_mean_confidence >= {confidence_threshold}`"


def _load_system_prompt_template() -> str:
    """Load the system prompt template from markdown file."""
    prompt_path = Path(__file__).parent / "system_prompt.md"
    return prompt_path.read_text()


def load_custom_system_prompt(path: str) -> str:
    """Load a custom system prompt from a file path."""
    return Path(path).read_text()


def get_system_prompt(
    env_name: str,
    confidence_threshold: float,
    method_description_path: str | None = None,
    target_model_description_path: str | None = None,
    goal_description_path: str | None = None,
) -> str:
    """Get the formatted system prompt for an environment."""
    defaults = DEFAULT_DESCRIPTION_PATHS[env_name]
    elicitation_type = "prefill" if "prefill" in env_name else "user_persona"

    method_description = _load_description(method_description_path or defaults["method"])
    setting_context = _load_description(target_model_description_path or defaults["target_model"])
    goal_description = _load_description(goal_description_path or defaults["goal"])

    template = _load_system_prompt_template()
    return template.format(
        method_name=METHOD_NAMES[env_name],
        elicitation_type=elicitation_type,
        setting_context=setting_context,
        method_description=method_description,
        goal_description=goal_description,
        tool_descriptions=get_tool_description(elicitation_type, confidence_threshold),
        success_criteria=get_success_criteria(confidence_threshold),
    )
