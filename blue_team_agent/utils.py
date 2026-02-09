"""Logging and serialization utilities for blue team sessions."""

import json
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import AssistantMessage, ResultMessage


class TraceLogger:
    """Handles trace file logging and human-readable output for agent sessions."""

    def __init__(self, trace_path: Path):
        self.trace_path = trace_path
        # Load existing entries if file exists (for resume)
        if trace_path.exists():
            with open(trace_path) as f:
                self.entries = json.load(f)
        else:
            self.entries = []

    def log(self, message) -> None:
        """Log a message to trace file and print human-readable output."""
        # Add to entries list
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": type(message).__name__,
            "content": serialize_message(message),
        }
        self.entries.append(trace_entry)

        # Write full JSON array with indentation
        with open(self.trace_path, "w") as f:
            json.dump(self.entries, f, indent=2)

        # Print human-readable output
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text)
                elif hasattr(block, "name"):
                    print(f"\n[Tool: {block.name}]")
                    # Print candidate elicitation input for evaluate_elicitation calls
                    if "evaluate_elicitation" in block.name and hasattr(block, "input"):
                        input_data = block.input
                        if "prefills" in input_data:
                            prefills = input_data["prefills"]
                            print(f"Prefills: {prefills}")
                        elif "persona_assistant" in input_data:
                            print(f"Persona Assistant: {input_data['persona_assistant']}")
                            print(f"Persona User: {input_data['persona_user']}")
        elif isinstance(message, ResultMessage):
            print(f"\n[Result: {message.subtype}]")

    def close(self) -> None:
        # Final write already happens in log(), nothing to close
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def print_session_summary(summary: dict) -> None:
    """Print human-readable session summary."""
    print(f"\n{'='*60}")
    print("Session Complete!")
    print(f"Total Evaluations: {summary['total_evaluations']}")
    print(f"Successful Elicitations: {len(summary['successful_elicitations'])}")
    if summary["best_score"]:
        print(f"Best Score (Confidence): {summary['best_score']}")
    print(f"{'='*60}\n")


def serialize_message(message) -> dict:
    """Serialize any SDK message type for logging."""
    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": [serialize_block(b) for b in message.content],
        }
    if isinstance(message, ResultMessage):
        return {
            "subtype": message.subtype,
            "result": str(message.result),
        }
    return {"raw": str(message)}


def serialize_block(block) -> dict:
    """Serialize content blocks for logging."""
    if hasattr(block, "text"):
        return {"type": "text", "text": block.text}
    if hasattr(block, "name"):
        return {"type": "tool_use", "name": block.name, "input": block.input}
    if hasattr(block, "tool_use_id"):
        return {"type": "tool_result", "tool_use_id": block.tool_use_id, "content": block.content}
    return {"type": "unknown", "raw": str(block)}


def write_summary(session_dir: Path) -> dict:
    """Generate session summary from evaluation log."""
    evaluation_log_path = session_dir / "evaluation_log.json"
    summary_path = session_dir / "summary.json"

    if not evaluation_log_path.exists():
        return {
            "total_evaluations": 0,
            "successful_elicitations": [],
            "best_score": None,
            "best_elicitation": None,
        }

    with open(evaluation_log_path) as f:
        entries = json.load(f)

    successful = [e for e in entries if e.get("is_success")]
    best = max(entries, key=lambda x: x["score"]) if entries else None

    summary = {
        "total_evaluations": len(entries),
        "successful_elicitations": [{"env_kwargs": e["env_kwargs"], "score": e["score"]} for e in successful],
        "best_score": best["score"] if best else None,
        "best_elicitation": best["env_kwargs"] if best else None,
        "final_timestamp": datetime.now().isoformat(),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
