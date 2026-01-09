"""
Paper Analysis Workflow.

Config-driven workflow dispatcher that loads workflow definitions from YAML files.

Usage:
    python -m plan_generator.workflow --paper-dir ./papers/my_paper

Workflows are defined in config/workflows/*.yaml
"""

import asyncio
import os
import sys
import argparse
from pathlib import Path

from mcp_agent.app import MCPApp

from .tracing import setup_tracing, workflow_span
from .llm_utils import get_tracing_config, setup_llm_api_keys
from .workflows.base import get_workflow_config
from .workflows.runner import create_workflow


_MODULE_DIR = Path(__file__).parent
_CONFIG_PATH = str(_MODULE_DIR / "mcp_agent.config.yaml")

app = MCPApp(name="paper_analyzer", settings=_CONFIG_PATH)


async def run_workflow(paper_dir: str, output_file: str = None):
    """Run the configured workflow and optionally save results."""
    setup_llm_api_keys()

    tracing_enabled, tracing_project, tracing_endpoint = get_tracing_config()
    setup_tracing(
        enabled=tracing_enabled,
        project_name=tracing_project,
        endpoint=tracing_endpoint
    )

    workflow_name = get_workflow_config()
    print(f"Running workflow: {workflow_name}")

    with workflow_span(f"{workflow_name}-workflow", paper_dir=paper_dir):
        async with app.run() as running_app:
            workflow = create_workflow(workflow_name)
            workflow_result = await workflow.run(paper_dir)
            result = workflow_result.value if hasattr(workflow_result, 'value') else workflow_result

        print("\n" + "=" * 80)
        print("=== RESULT ===")
        print("=" * 80)
        print(result)
        print("=" * 80)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"Result saved to {output_file}")

        return result


async def run_workflow_batch(papers: list[tuple[str, str]]):
    """Run workflow for multiple papers concurrently within a single app context.

    Args:
        papers: List of (paper_dir, output_file) tuples

    Returns:
        List of (paper_dir, success, result_or_error) tuples
    """
    setup_llm_api_keys()

    tracing_enabled, tracing_project, tracing_endpoint = get_tracing_config()
    setup_tracing(
        enabled=tracing_enabled,
        project_name=tracing_project,
        endpoint=tracing_endpoint
    )

    workflow_name = get_workflow_config()
    print(f"Running workflow: {workflow_name}")

    async def process_single(paper_dir: str, output_file: str):
        """Process a single paper within the shared app context."""
        with workflow_span(f"{workflow_name}-workflow", paper_dir=paper_dir):
            try:
                workflow = create_workflow(workflow_name)
                workflow_result = await workflow.run(paper_dir)
                result = workflow_result.value if hasattr(workflow_result, 'value') else workflow_result

                print(f"\nWorkflow completed (length: {len(result)} chars)")

                if output_file:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(result)
                    print(f"Result saved to {output_file}")

                return paper_dir, True, result
            except Exception as e:
                print(f"Error processing {paper_dir}: {e}")
                return paper_dir, False, str(e)

    async with app.run() as running_app:
        tasks = [process_single(paper_dir, output_file) for paper_dir, output_file in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions from gather itself
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                paper_dir = papers[i][0]
                processed_results.append((paper_dir, False, str(result)))
            else:
                processed_results.append(result)

        return processed_results


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser(description="Analyze research paper using configured workflow")
    parser.add_argument("--paper-dir", required=True, help="Directory containing paper page files")
    parser.add_argument("--output", "-o", help="Output file path")
    args = parser.parse_args()

    if not os.path.isdir(args.paper_dir):
        print(f"Error: Directory does not exist: {args.paper_dir}")
        sys.exit(1)

    page_files = [f for f in os.listdir(args.paper_dir) if f.endswith('.txt') and '_page_' in f]
    if not page_files:
        print(f"Error: No page files (*_page_N.txt) found in: {args.paper_dir}")
        sys.exit(1)

    print(f"Found {len(page_files)} page file(s)")
    asyncio.run(run_workflow(args.paper_dir, args.output))
