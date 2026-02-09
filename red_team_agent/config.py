"""Configuration for the red team agent."""

from dataclasses import dataclass, field
from pathlib import Path


# Human-readable method names for each environment
METHOD_NAMES = {
    # SSC environments
    "ssc_prefill": "Prefill",
    "ssc_user_persona": "User Persona",
    "ssc_act_tokens": "Activation Tokens",
    # User gender environments (Gemma)
    "user_gender_prefill": "Prefill",
    "user_gender_user_persona": "User Persona",
    "user_gender_sae_desc": "SAE Descriptions",
    # User gender environments (Qwen)
    "user_gender_prefill_qwen": "Prefill",
    "user_gender_user_persona_qwen": "User Persona",
}


@dataclass
class RedTeamConfig:
    """Configuration for a red team session."""

    env_name: str  # Environment name (e.g., "ssc_prefill", "user_gender_prefill")
    session_id: str  # Unique session identifier
    audit_threshold: float  # audit_accuracy must be below this
    internalize_baseline: float  # baseline internalization accuracy
    internalize_threshold: float  # threshold = baseline * factor (success criteria)
    max_evaluations: int | None = None  # Max evaluations (None = unlimited)
    eval_server_url: str = "http://localhost:8000"  # Evaluation server URL
    results_dir: Path = field(default_factory=lambda: Path("results/red_team_sessions"))
    audit_only: bool = False  # If True, skip internalization scoring
    show_count: bool = False  # If True, show evaluation count to agent

    @property
    def session_dir(self) -> Path:
        """Get the session directory path."""
        return self.results_dir / self.session_id

    @property
    def method_name(self) -> str:
        """Get human-readable method name."""
        return METHOD_NAMES.get(self.env_name, self.env_name)

    def to_dict(self) -> dict:
        """Serialize config to a dictionary for logging."""
        return {
            "env_name": self.env_name,
            "session_id": self.session_id,
            "max_evaluations": self.max_evaluations,
            "eval_server_url": self.eval_server_url,
            "results_dir": str(self.results_dir),
            "audit_only": self.audit_only,
            "show_count": self.show_count,
            "audit_threshold": self.audit_threshold,
            "internalize_baseline": self.internalize_baseline,
            "internalize_threshold": self.internalize_threshold,
        }
