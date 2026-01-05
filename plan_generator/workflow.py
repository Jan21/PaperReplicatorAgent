"""
Code Analysis Workflow.

This module defines the code analysis workflow that runs on either
asyncio or Temporal depending on the execution_engine config setting.

Usage:
    python -m plan_generator.workflow --paper-dir ./papers/my_paper

Or programmatically:
    from plan_generator.workflow import app, main
    import asyncio
    asyncio.run(main("/path/to/paper_dir"))
"""

import os
import sys
import argparse
from pathlib import Path

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.executor.workflow import Workflow, WorkflowResult
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from .tracing import setup_tracing, workflow_span
from .prompts import (
    PAPER_ALGORITHM_ANALYSIS_PROMPT,
    PAPER_CONCEPT_ANALYSIS_PROMPT,
    CODE_PLANNING_PROMPT,
)
from .llm_utils import get_token_limits, get_llm_params, get_tracing_config, get_agent_servers


# Get the path to the config file in this module's directory
_MODULE_DIR = Path(__file__).parent
_CONFIG_PATH = str(_MODULE_DIR / "mcp_agent.config.yaml")

# Create the MCPApp instance with explicit config path
# This works with both asyncio and temporal execution engines
app = MCPApp(name="code_analyzer", settings=_CONFIG_PATH)


@app.workflow
class AnalyzePaperWorkflow(Workflow[str]):
    """
    Workflow for analyzing research papers and generating implementation plans.

    This workflow orchestrates three specialized agents:
    - ConceptAnalysisAgent: Analyzes system architecture and conceptual framework
    - AlgorithmAnalysisAgent: Extracts algorithms, formulas, and technical details
    - CodePlannerAgent: Integrates outputs into a comprehensive implementation plan
    """

    @app.workflow_run
    async def run(self, paper_dir: str) -> WorkflowResult[str]:
        """
        Execute the code analysis workflow.

        Args:
            paper_dir: Directory path containing the research paper (.md file)

        Returns:
            WorkflowResult containing the YAML implementation plan
        """
        print(f"Code analysis workflow started")
        print(f"   Paper directory: {paper_dir}")

        # STEP 1: Read paper file
        paper_content = None
        paper_file_path = None

        try:
            for filename in os.listdir(paper_dir):
                if filename.endswith(".md"):
                    paper_file_path = os.path.join(paper_dir, filename)
                    with open(paper_file_path, "r", encoding="utf-8") as f:
                        paper_content = f.read()
                    print(f"Paper file loaded: {paper_file_path} ({len(paper_content)} chars)")
                    break

            if not paper_content:
                return WorkflowResult(value="No paper file found")
        except Exception as e:
            print(f"Error reading paper file: {e}")
            return WorkflowResult(value=f"Error reading paper: {e}")

        # STEP 2: Configure agents from config
        servers = get_agent_servers()
        concept_servers = servers["concept_analysis"]
        algorithm_servers = servers["algorithm_analysis"]
        planner_servers = servers["code_planner"]

        print(f"   Agent configurations:")
        print(f"     ConceptAnalysis: {concept_servers or 'no tools'}")
        print(f"     AlgorithmAnalysis: {algorithm_servers or 'no tools'}")
        print(f"     CodePlanner: {planner_servers or 'no tools'}")

        concept_analysis_agent = Agent(
            name="ConceptAnalysisAgent",
            instruction=PAPER_CONCEPT_ANALYSIS_PROMPT,
            server_names=concept_servers,
        )
        algorithm_analysis_agent = Agent(
            name="AlgorithmAnalysisAgent",
            instruction=PAPER_ALGORITHM_ANALYSIS_PROMPT,
            server_names=algorithm_servers,
        )
        code_planner_agent = Agent(
            name="CodePlannerAgent",
            instruction=CODE_PLANNING_PROMPT,
            server_names=planner_servers,
        )

        code_aggregator_agent = ParallelLLM(
            fan_in_agent=code_planner_agent,
            fan_out_agents=[concept_analysis_agent, algorithm_analysis_agent],
            llm_factory=OpenAIAugmentedLLM,
        )

        base_max_tokens, _ = get_token_limits()
        temperature, max_iterations = get_llm_params()

        # STEP 3: Configure request parameters
        enhanced_params = RequestParams(
            maxTokens=base_max_tokens,
            temperature=temperature,
            max_iterations=max_iterations,
        )

        # STEP 4: Construct message with paper content
        message = f"""Analyze the research paper provided below. The paper file has been pre-loaded for you.

=== PAPER CONTENT START ===
{paper_content}
=== PAPER CONTENT END ===

Based on this paper, generate a comprehensive code reproduction plan that includes:

1. Complete system architecture and component breakdown
2. All algorithms, formulas, and implementation details
3. Detailed file structure and implementation roadmap

You may use web search (brave_web_search) if you need clarification on algorithms, methods, or concepts.

The goal is to create a reproduction plan detailed enough for independent implementation."""

        # STEP 5: Execute the analysis
        print("Executing code analysis via ParallelLLM...")
        result = await code_aggregator_agent.generate_str(
            message=message, request_params=enhanced_params
        )

        print(f"Code analysis completed (length: {len(result)} chars)")
        return WorkflowResult(value=result)


async def run_workflow(paper_dir: str, output_file: str = None):
    """
    Run the analysis workflow and optionally save results.

    Works with both asyncio and temporal execution engines automatically.
    """
    # Initialize Phoenix tracing from config
    tracing_enabled, tracing_project, tracing_endpoint = get_tracing_config()
    setup_tracing(
        enabled=tracing_enabled,
        project_name=tracing_project,
        endpoint=tracing_endpoint
    )

    with workflow_span("analyze-paper-workflow", paper_dir=paper_dir):
        async with app.run() as running_app:
            executor = running_app.executor

            # Check if we're using Temporal or asyncio executor
            if hasattr(executor, 'execute_workflow'):
                # Temporal executor - use execute_workflow
                result = await executor.execute_workflow(
                    "AnalyzePaperWorkflow",
                    paper_dir
                )
            else:
                # Asyncio executor - instantiate workflow and run directly
                workflow = AnalyzePaperWorkflow()
                workflow_result = await workflow.run(paper_dir)
                result = workflow_result.value if hasattr(workflow_result, 'value') else workflow_result

        print("\n" + "=" * 80)
        print("=== ANALYSIS RESULT ===")
        print("=" * 80)
        print(result)
        print("=" * 80)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"Result saved to {output_file}")

        return result


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser(description="Analyze research paper and generate implementation plan")
    parser.add_argument("--paper-dir", required=True, help="Directory containing the paper (.md file)")
    parser.add_argument("--output", "-o", help="Output file path for the result")
    args = parser.parse_args()

    # Verify directory exists
    if not os.path.isdir(args.paper_dir):
        print(f"Error: Directory {args.paper_dir} does not exist!")
        sys.exit(1)

    # Verify it contains .md file
    md_files = [f for f in os.listdir(args.paper_dir) if f.endswith('.md')]
    if not md_files:
        print(f"Error: No .md file found in directory {args.paper_dir}!")
        sys.exit(1)

    print(f"Found .md files: {md_files}")

    asyncio.run(run_workflow(args.paper_dir, args.output))
