"""Config-driven workflow runner."""

from mcp_agent.agents.agent import Agent
from mcp_agent.executor.workflow import Workflow, WorkflowResult
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM

from ..config import load_workflow_config, WorkflowConfig
from ..llm_utils import get_token_limits, get_llm_params, get_llm_class
from .base import load_paper_content


class ConfigDrivenWorkflow(Workflow[str]):
    """A workflow that runs based on YAML configuration."""

    def __init__(self, config: WorkflowConfig):
        self.config = config

    async def run(self, paper_dir: str) -> WorkflowResult[str]:
        """Execute the workflow based on its configuration."""
        print(f"{self.config.name} workflow started")
        print(f"   Paper directory: {paper_dir}")

        paper_content = load_paper_content(paper_dir)
        if not paper_content:
            return WorkflowResult(value="No paper page files found")

        base_max_tokens, _ = get_token_limits()
        temperature, max_iterations = get_llm_params()

        params = RequestParams(
            maxTokens=base_max_tokens,
            temperature=temperature,
            max_iterations=max_iterations,
        )

        message = f"""Analyze the research paper provided below.

=== PAPER CONTENT START ===
{paper_content}
=== PAPER CONTENT END ===
"""

        if self.config.type == "single":
            result = await self._run_single(message, params)
        else:
            result = await self._run_parallel(message, params)

        print(f"Workflow completed (length: {len(result)} chars)")
        return WorkflowResult(value=result)

    async def _run_single(self, message: str, params: RequestParams) -> str:
        """Run a single-agent workflow."""
        agent_config = self.config.agents[0]

        agent = Agent(
            name=agent_config.name,
            instruction=agent_config.get_prompt(),
            server_names=agent_config.tools,
        )

        print(f"   Agent: {agent_config.name} (tools: {agent_config.tools or 'none'})")

        llm_class = get_llm_class()
        llm = llm_class(agent=agent)

        return await llm.generate_str(message=message, request_params=params)

    async def _run_parallel(self, message: str, params: RequestParams) -> str:
        """Run a parallel fan-out/fan-in workflow."""
        fan_out_agents = []
        for agent_config in self.config.agents:
            agent = Agent(
                name=agent_config.name,
                instruction=agent_config.get_prompt(),
                server_names=agent_config.tools,
            )
            fan_out_agents.append(agent)
            print(f"   Fan-out: {agent_config.name} (tools: {agent_config.tools or 'none'})")

        fan_in_config = self.config.fan_in_agent
        fan_in_agent = Agent(
            name=fan_in_config.name,
            instruction=fan_in_config.get_prompt(),
            server_names=fan_in_config.tools,
        )
        print(f"   Fan-in: {fan_in_config.name} (tools: {fan_in_config.tools or 'none'})")

        parallel_llm = ParallelLLM(
            fan_in_agent=fan_in_agent,
            fan_out_agents=fan_out_agents,
            llm_factory=get_llm_class(),
        )

        return await parallel_llm.generate_str(message=message, request_params=params)


def create_workflow(workflow_name: str) -> ConfigDrivenWorkflow:
    """Create a workflow instance from configuration."""
    config = load_workflow_config(workflow_name)
    return ConfigDrivenWorkflow(config)
