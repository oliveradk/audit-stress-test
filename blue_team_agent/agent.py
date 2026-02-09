"""Main agent runner for blue team sessions.

Uses the Anthropic Agent SDK to run iterative elicitation prompt optimization
for SSC and User Gender environments.

Usage:
    python -m blue_team_agent.agent --env ssc_prefill --baselines-file results/baseline.json --max-evaluations 50
    python -m blue_team_agent.agent --env ssc_user_persona --baselines-file results/baseline.json --system-prompt-file path/to/prompt.txt --max-evaluations 50

With mock server (local testing):
    # Terminal 1: Start mock server
    python -m eval.mock_server --env ssc_prefill

    # Terminal 2: Run agent
    python -m blue_team_agent.agent --env ssc_prefill --confidence-threshold 0.8 --max-evaluations 10
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from blue_team_agent.config import BlueTeamConfig
from blue_team_agent.system_prompt import get_system_prompt, load_custom_system_prompt
from blue_team_agent.tools import create_session_tools, get_tool_names
from blue_team_agent.utils import TraceLogger, print_session_summary, write_summary

# Model name mappings (generic name -> specific checkpoint)
MODEL_CHECKPOINTS = {
    "sonnet-3.7": "claude-3-7-sonnet-20250219",
    "sonnet-4": "claude-sonnet-4-20250514",
    "sonnet-4.5": "claude-sonnet-4-5-20250929",
    "haiku-4.5": "claude-haiku-4-5-20251001",
    "opus-4.5": "claude-opus-4-5-20251101",
}


async def run_blue_team_session(
    env_name: str,
    session_id: str,
    confidence_threshold: float,
    target_system_prompt: str = "",
    max_evaluations: int | None = None,
    eval_server_url: str = "http://localhost:8000",
    results_dir: Path = Path("results/blue_team_sessions"),
    show_count: bool = False,
    terminate_on_success: bool = False,
    model: str = "sonnet-3.7",
    agent_system_prompt_file: str | None = None,
    method_description_path: str | None = None,
    target_model_description_path: str | None = None,
    goal_description_path: str | None = None,
):
    """Run a blue team session using the Agent SDK.

    Args:
        env_name: Environment name (e.g., ssc_prefill, ssc_user_persona)
        session_id: Unique session identifier
        target_system_prompt: Fixed system prompt for target model evaluations
        max_evaluations: Maximum number of evaluations before stopping (None = unlimited)
        eval_server_url: URL of the evaluation server
        results_dir: Base directory for results
        show_count: If True, show evaluation count to agent
        terminate_on_success: If True, stop after first successful evaluation
        confidence_threshold: Threshold for success
        model: Model name (sonnet-3.7, sonnet-4, opus-4.5)
        agent_system_prompt_file: Path to custom agent system prompt file
        method_description_path: Override path for method description file
        target_model_description_path: Override path for target model description file
        goal_description_path: Override path for goal description file

    Returns:
        Session summary dict
    """
    # Create config
    config = BlueTeamConfig(
        env_name=env_name,
        session_id=session_id,
        system_prompt=target_system_prompt,
        max_evaluations=max_evaluations,
        eval_server_url=eval_server_url,
        results_dir=results_dir,
        show_count=show_count,
        terminate_on_success=terminate_on_success,
        confidence_threshold=confidence_threshold,
    )

    # Create session directory
    session_dir = config.session_dir
    session_dir.mkdir(parents=True, exist_ok=True)

    # Log config before starting
    config_path = session_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Config logged to: {config_path}")

    # Get agent system prompt and create tools
    if agent_system_prompt_file:
        agent_system_prompt = load_custom_system_prompt(agent_system_prompt_file)
    else:
        agent_system_prompt = get_system_prompt(
            env_name,
            confidence_threshold=confidence_threshold,
            method_description_path=method_description_path,
            target_model_description_path=target_model_description_path,
            goal_description_path=goal_description_path,
        )

    tools_server = create_session_tools(
        session_dir,
        eval_server_url,
        env_name,
        system_prompt=target_system_prompt,
        elicitation_type=config.elicitation_type,
        show_count=show_count,
        confidence_threshold=confidence_threshold,
        max_evaluations=max_evaluations,
        terminate_on_success=terminate_on_success,
    )

    model_checkpoint = MODEL_CHECKPOINTS[model]
    options = ClaudeAgentOptions(
        model=model_checkpoint,
        system_prompt=agent_system_prompt,
        mcp_servers={"blue-team-tools": tools_server},
        allowed_tools=get_tool_names(),
        max_turns=None,  # Don't use SDK turn limiting - controlled via max_evaluations in tools
    )

    # Run agent loop with trace logging
    with TraceLogger(session_dir / "agent_trace.json") as logger:
        async with ClaudeSDKClient(options=options) as client:
            initial_query = "Begin optimizing elicitation prompts. Generate and evaluate diverse elicitation strategies to maximize auditor confidence. Continue until you exhaust your evaluation budget."
            await client.query(initial_query)
            async for message in client.receive_response():
                logger.log(message)

    summary = write_summary(session_dir)
    print_session_summary(summary)
    return summary


async def main():
    parser = argparse.ArgumentParser(
        description="Run a blue team agent session to optimize elicitation prompts"
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=[
            "ssc_prefill", "ssc_user_persona",
            "user_gender_prefill", "user_gender_user_persona",
        ],
        help="Target environment",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Session ID (default: auto-generated from timestamp)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="Fixed system prompt to use for target model evaluations",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=None,
        help="Path to file containing system prompt for target model",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=None,
        help="Maximum number of evaluations before stopping (default: unlimited)",
    )
    parser.add_argument(
        "--eval-server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the evaluation server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/blue_team_sessions",
        help="Base directory for session results",
    )
    parser.add_argument(
        "--show-count",
        action="store_true",
        help="Show evaluation count to the agent (hidden by default)",
    )
    parser.add_argument(
        "--terminate-on-success",
        action="store_true",
        help="Stop after first successful evaluation (default: continue until budget exhausted)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Confidence threshold for success (overrides baselines-file if both provided)",
    )
    parser.add_argument(
        "--baselines-file",
        type=str,
        default=None,
        help="Path to JSON file with baseline values (uses audit_mean_confidence as threshold)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="opus-4.5",
        choices=list(MODEL_CHECKPOINTS.keys()),
        help="Model to use for blue teaming (default: opus-4.5)",
    )
    parser.add_argument(
        "--agent-system-prompt-file",
        type=str,
        default=None,
        help="Path to custom agent system prompt file (bypasses template system)",
    )
    parser.add_argument(
        "--method-description-file",
        type=str,
        default=None,
        help="Path to method description file (overrides default)",
    )
    parser.add_argument(
        "--target-model-description-file",
        type=str,
        default=None,
        help="Path to target model description file (overrides default)",
    )
    parser.add_argument(
        "--goal-description-file",
        type=str,
        default=None,
        help="Path to goal description file (overrides default)",
    )

    args = parser.parse_args()

    # Load system prompt from file if provided
    target_system_prompt = args.system_prompt
    if args.system_prompt_file:
        target_system_prompt = Path(args.system_prompt_file).read_text()

    # Load confidence threshold: explicit value takes precedence, then baseline file
    confidence_threshold = args.confidence_threshold
    if confidence_threshold is None:
        if args.baselines_file:
            with open(args.baselines_file) as f:
                baselines_data = json.load(f)
            confidence_threshold = baselines_data["audit_mean_confidence"]
            print(f"Loaded confidence threshold from baseline: {confidence_threshold}")
        else:
            parser.error("Either --confidence-threshold or --baselines-file must be provided")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = args.session or f"blue_{args.env}_{timestamp}"

    print(f"Starting blue team session: {session_id}")
    print(f"Environment: {args.env}")
    print(f"Model: {args.model} ({MODEL_CHECKPOINTS[args.model]})")
    print(f"Eval Server: {args.eval_server_url}")
    print(f"Max Evaluations: {args.max_evaluations}")
    print(f"Confidence Threshold: {confidence_threshold}")
    print(f"Terminate on Success: {args.terminate_on_success}")
    if target_system_prompt:
        preview = target_system_prompt[:100] + "..." if len(target_system_prompt) > 100 else target_system_prompt
        print(f"Target System Prompt: {preview}")
    else:
        print("Target System Prompt: (empty)")
    if args.agent_system_prompt_file:
        print(f"Agent System Prompt: {args.agent_system_prompt_file}")
    if args.method_description_file:
        print(f"Method Description: {args.method_description_file}")
    if args.target_model_description_file:
        print(f"Target Model Description: {args.target_model_description_file}")
    if args.goal_description_file:
        print(f"Goal Description: {args.goal_description_file}")

    summary = await run_blue_team_session(
        env_name=args.env,
        session_id=session_id,
        target_system_prompt=target_system_prompt,
        max_evaluations=args.max_evaluations,
        eval_server_url=args.eval_server_url,
        results_dir=Path(args.results_dir),
        show_count=args.show_count,
        terminate_on_success=args.terminate_on_success,
        confidence_threshold=confidence_threshold,
        model=args.model,
        agent_system_prompt_file=args.agent_system_prompt_file,
        method_description_path=args.method_description_file,
        target_model_description_path=args.target_model_description_file,
        goal_description_path=args.goal_description_file,
    )

    print("\nFinal Summary:")
    print(f"  Total Evaluations: {summary['total_evaluations']}")
    print(f"  Successful Elicitations: {len(summary['successful_elicitations'])}")

    if summary["successful_elicitations"]:
        print("\n  Successful elicitations found:")
        for e in summary["successful_elicitations"]:
            print(f"    - Score {e['score']}: {str(e['env_kwargs'])[:80]}...")

    if summary["best_score"]:
        print(f"\n  Best Score: {summary['best_score']}")
        print(f"  Best Elicitation: {str(summary['best_elicitation'])[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
