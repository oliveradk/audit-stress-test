"""Configuration for the blue team agent."""

from dataclasses import dataclass, field
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


@dataclass
class BlueTeamConfig:
    """Configuration for a blue team session."""

    env_name: str  # Environment name (e.g., "ssc_prefill", "ssc_user_persona")
    session_id: str  # Unique session identifier
    confidence_threshold: float  # audit_mean_confidence must be above this
    system_prompt: str = ""  # Fixed system prompt (from red team)
    max_evaluations: int | None = None  # Max evaluations (None = unlimited)
    eval_server_url: str = "http://localhost:8000"  # Evaluation server URL
    results_dir: Path = field(default_factory=lambda: Path("results/blue_team_sessions"))
    show_count: bool = False  # If True, show evaluation count to agent
    terminate_on_success: bool = False  # If True, stop after first successful evaluation

    @property
    def session_dir(self) -> Path:
        """Get the session directory path."""
        return self.results_dir / self.session_id

    @property
    def method_name(self) -> str:
        """Get human-readable method name."""
        return METHOD_NAMES.get(self.env_name, self.env_name)

    @property
    def elicitation_type(self) -> str:
        """Return 'prefill' or 'user_persona' based on env."""
        if "prefill" in self.env_name:
            return "prefill"
        elif "user_persona" in self.env_name:
            return "user_persona"
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")

    def to_dict(self) -> dict:
        """Serialize config to a dictionary for logging."""
        return {
            "env_name": self.env_name,
            "session_id": self.session_id,
            "system_prompt": self.system_prompt,
            "max_evaluations": self.max_evaluations,
            "eval_server_url": self.eval_server_url,
            "results_dir": str(self.results_dir),
            "show_count": self.show_count,
            "confidence_threshold": self.confidence_threshold,
            "elicitation_type": self.elicitation_type,
            "terminate_on_success": self.terminate_on_success,
        }
