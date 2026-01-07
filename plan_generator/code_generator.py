"""
Code Generation Workflow.

This module generates actual code files from an implementation plan.
It processes the plan in batches: core → support → experiment → docs.

Usage:
    from plan_generator.code_generator import run_code_generation
    import asyncio
    asyncio.run(run_code_generation("analysis_result.yaml", "my_project"))

Or via command line:
    python -m plan_generator.code_generator --plan analysis_result.yaml --project my_project
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.executor.workflow import Workflow, WorkflowResult
from mcp_agent.workflows.llm.augmented_llm import RequestParams

from .tracing import setup_tracing, workflow_span
from .prompts import get_code_generation_prompt
from .llm_utils import (
    get_token_limits,
    get_llm_params,
    get_tracing_config,
    get_llm_class,
    setup_llm_api_keys,
)


# Module directory and config
_MODULE_DIR = Path(__file__).parent
_CONFIG_PATH = str(_MODULE_DIR / "mcp_agent.config.yaml")
_PROJECTS_DIR = _MODULE_DIR.parent / "projects"

# Create MCPApp instance
code_gen_app = MCPApp(name="code_generator", settings=_CONFIG_PATH)


def parse_generated_files(response: str) -> List[Tuple[str, str]]:
    """
    Parse LLM response to extract file paths and contents.

    Expects format:
        ===FILE: path/to/file.py===
        <contents>
        ===END_FILE===

    Args:
        response: Raw LLM response string

    Returns:
        List of (file_path, file_content) tuples
    """
    files = []
    # Pattern to match file blocks
    pattern = r'===FILE:\s*(.+?)===\s*\n(.*?)===END_FILE==='
    matches = re.findall(pattern, response, re.DOTALL)

    for file_path, content in matches:
        # Clean up the file path and content
        file_path = file_path.strip()
        content = content.strip()
        if file_path and content:
            files.append((file_path, content))

    return files


def write_files_to_project(files: List[Tuple[str, str]], project_dir: Path) -> List[str]:
    """
    Write generated files to the project directory.

    Args:
        files: List of (relative_path, content) tuples
        project_dir: Base directory for the project

    Returns:
        List of successfully written file paths
    """
    written_files = []

    for relative_path, content in files:
        # Construct full path
        file_path = project_dir / relative_path

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            written_files.append(str(file_path))
            print(f"  Written: {relative_path}")
        except Exception as e:
            print(f"  Error writing {relative_path}: {e}")

    return written_files


def categorize_files_from_plan(plan_content: str) -> dict:
    """
    Analyze the plan to categorize files into generation batches.

    This is a heuristic categorization based on common patterns.
    Returns dict with keys: core, support, experiment, docs
    """
    categories = {
        "core": [],
        "support": [],
        "experiment": [],
        "docs": []
    }

    # Extract file structure section from plan
    file_section_match = re.search(
        r'file_structure:\s*\|?\s*\n(.*?)(?=\n\s*\w+:|$)',
        plan_content,
        re.DOTALL
    )

    if not file_section_match:
        # Try alternative format
        file_section_match = re.search(
            r'file_structure:(.*?)(?=implementation_components:|validation_approach:|$)',
            plan_content,
            re.DOTALL
        )

    if file_section_match:
        file_section = file_section_match.group(1)
        # Extract file paths (look for .py, .txt, .md, .yaml, etc.)
        file_pattern = r'[\w/\-_]+\.(?:py|txt|md|yaml|yml|json|sh)'
        files = re.findall(file_pattern, file_section)

        for f in files:
            f_lower = f.lower()
            if any(x in f_lower for x in ['readme', 'requirements', 'license', 'contributing']):
                categories["docs"].append(f)
            elif any(x in f_lower for x in ['train', 'run', 'main', 'experiment', 'eval', 'test_', 'demo']):
                categories["experiment"].append(f)
            elif any(x in f_lower for x in ['util', 'helper', 'common', 'config', 'data_', 'preprocess']):
                categories["support"].append(f)
            else:
                categories["core"].append(f)

    return categories


@code_gen_app.workflow
class GenerateCodeWorkflow(Workflow[str]):
    """
    Workflow for generating code files from an implementation plan.

    Generates code in batches:
    1. Core files (algorithms, models)
    2. Support files (utilities, data handling)
    3. Experiment files (training, evaluation scripts)
    4. Documentation files (README, requirements)
    """

    @code_gen_app.workflow_run
    async def run(self, plan_content: str, project_name: str) -> WorkflowResult[str]:
        """
        Execute the code generation workflow.

        Args:
            plan_content: The YAML implementation plan content
            project_name: Name for the output project directory

        Returns:
            WorkflowResult with summary of generated files
        """
        print(f"Code generation workflow started")
        print(f"   Project name: {project_name}")

        # Setup project directory
        project_dir = _PROJECTS_DIR / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Output directory: {project_dir}")

        # Categorize files from plan
        file_categories = categorize_files_from_plan(plan_content)
        print(f"   File categories: {', '.join(f'{k}: {len(v)}' for k, v in file_categories.items())}")

        # Get LLM parameters
        base_max_tokens, _ = get_token_limits()
        temperature, max_iterations = get_llm_params()

        params = RequestParams(
            maxTokens=base_max_tokens,
            temperature=temperature,
            max_iterations=max_iterations,
        )

        all_written_files = []
        generation_summary = []

        # Generate code in batches
        batch_order = ["core", "support", "experiment", "docs"]

        for batch_type in batch_order:
            print(f"\n{'='*60}")
            print(f"Generating {batch_type.upper()} files...")
            print(f"{'='*60}")

            # Create agent with batch-specific prompt
            code_gen_agent = Agent(
                name=f"CodeGenerator_{batch_type}",
                instruction=get_code_generation_prompt(batch_type),
                server_names=[],  # No tools needed for pure generation
            )

            # Get LLM class and create augmented LLM
            llm_class = get_llm_class()
            llm = llm_class(agent=code_gen_agent)

            # Build context with previously generated files
            previous_files_context = ""
            if all_written_files:
                previous_files_context = f"""
## Previously Generated Files
The following files have already been generated in this project:
{chr(10).join(f'- {f}' for f in all_written_files)}

Ensure your new files integrate properly with these existing files.
"""

            # Build message
            message = f"""# Implementation Plan

{plan_content}

{previous_files_context}

# Files to Generate in This Batch ({batch_type})

Based on the implementation plan above, generate the {batch_type} files.
Focus on the files that belong to this category:
{chr(10).join(f'- {f}' for f in file_categories.get(batch_type, [])) or 'Determine appropriate files from the plan'}

Generate complete, working implementations for each file.
Use the ===FILE: path=== and ===END_FILE=== markers as specified."""

            # Generate code
            try:
                result = await llm.generate_str(
                    message=message,
                    request_params=params
                )

                # Parse and write files
                files = parse_generated_files(result)

                if files:
                    written = write_files_to_project(files, project_dir)
                    all_written_files.extend(written)
                    generation_summary.append(f"{batch_type}: {len(files)} files generated")
                else:
                    print(f"  No files parsed from {batch_type} batch")
                    generation_summary.append(f"{batch_type}: 0 files (parsing issue)")

            except Exception as e:
                print(f"  Error in {batch_type} batch: {e}")
                generation_summary.append(f"{batch_type}: error - {str(e)[:50]}")

        # Create summary
        summary = f"""
Code Generation Complete
========================
Project: {project_name}
Location: {project_dir}

Generated Files ({len(all_written_files)} total):
{chr(10).join(f'  - {f}' for f in all_written_files)}

Batch Summary:
{chr(10).join(f'  - {s}' for s in generation_summary)}
"""
        print(summary)
        return WorkflowResult(value=summary)


async def run_code_generation(
    plan_file: str,
    project_name: Optional[str] = None
) -> str:
    """
    Run the code generation workflow from a plan file.

    Args:
        plan_file: Path to the YAML implementation plan
        project_name: Name for the output project (defaults to plan filename)

    Returns:
        Summary of generated files
    """
    # Setup API keys
    setup_llm_api_keys()

    # Setup tracing
    tracing_enabled, tracing_project, tracing_endpoint = get_tracing_config()
    setup_tracing(
        enabled=tracing_enabled,
        project_name=tracing_project,
        endpoint=tracing_endpoint
    )

    # Read plan file
    plan_path = Path(plan_file)
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_file}")

    with open(plan_path, 'r', encoding='utf-8') as f:
        plan_content = f.read()

    # Determine project name
    if not project_name:
        project_name = plan_path.stem.replace('_result', '').replace('analysis', 'project')

    with workflow_span("code-generation-workflow", project_name=project_name):
        async with code_gen_app.run() as running_app:
            executor = running_app.executor

            if hasattr(executor, 'execute_workflow'):
                # Temporal executor
                result = await executor.execute_workflow(
                    "GenerateCodeWorkflow",
                    plan_content,
                    project_name
                )
            else:
                # Asyncio executor
                workflow = GenerateCodeWorkflow()
                workflow_result = await workflow.run(plan_content, project_name)
                result = workflow_result.value if hasattr(workflow_result, 'value') else workflow_result

    return result


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser(
        description="Generate code files from an implementation plan"
    )
    parser.add_argument(
        "--plan", "-p",
        required=True,
        help="Path to the YAML implementation plan file"
    )
    parser.add_argument(
        "--project", "-n",
        help="Name for the output project directory"
    )
    args = parser.parse_args()

    # Verify plan file exists
    if not os.path.isfile(args.plan):
        print(f"Error: Plan file not found: {args.plan}")
        exit(1)

    asyncio.run(run_code_generation(args.plan, args.project))
