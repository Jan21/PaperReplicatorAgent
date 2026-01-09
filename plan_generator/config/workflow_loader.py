"""Workflow configuration loader."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    name: str
    prompt: str  # Prompt file name (e.g., "experiments" -> prompts/experiments.txt)
    tools: list[str] = field(default_factory=list)

    def get_prompt(self) -> str:
        """Load and return the prompt string from the .txt file."""
        prompts_dir = Path(__file__).parent.parent / "prompts"
        prompt_path = prompts_dir / f"{self.prompt}.txt"

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        return prompt_path.read_text(encoding="utf-8")


@dataclass
class WorkflowConfig:
    """Configuration for a workflow."""
    name: str
    type: str  # "single" or "parallel"
    agents: list[AgentConfig] = field(default_factory=list)
    fan_in_agent: AgentConfig | None = None

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "WorkflowConfig":
        """Create WorkflowConfig from a dictionary."""
        workflow_type = data.get("type", "single")

        if workflow_type == "single":
            agent_data = data.get("agent", {})
            agents = [AgentConfig(
                name=agent_data.get("name", "Agent"),
                prompt=agent_data.get("prompt"),
                tools=agent_data.get("tools", []),
            )]
            return cls(name=name, type=workflow_type, agents=agents)

        elif workflow_type == "parallel":
            fan_out = data.get("fan_out_agents", [])
            fan_in = data.get("fan_in_agent", {})

            agents = [
                AgentConfig(
                    name=a.get("name", f"Agent{i}"),
                    prompt=a.get("prompt"),
                    tools=a.get("tools", []),
                )
                for i, a in enumerate(fan_out)
            ]

            fan_in_agent = AgentConfig(
                name=fan_in.get("name", "AggregatorAgent"),
                prompt=fan_in.get("prompt"),
                tools=fan_in.get("tools", []),
            )

            return cls(name=name, type=workflow_type, agents=agents, fan_in_agent=fan_in_agent)

        raise ValueError(f"Unknown workflow type: {workflow_type}")


def load_workflow_config(workflow_name: str) -> WorkflowConfig:
    """Load a workflow configuration by name."""
    config_dir = Path(__file__).parent / "workflows"
    config_path = config_dir / f"{workflow_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Workflow config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return WorkflowConfig.from_dict(workflow_name, data)
