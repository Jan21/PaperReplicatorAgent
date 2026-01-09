#!/usr/bin/env python3
"""
Structure Extraction Workflow.

Processes all papers in the papers folder in parallel, extracts sections,
and calls LLM for each section to generate structured analysis.

Usage:
    python -m plan_generator.structure_extraction.extract_structure \
        --papers-dir ./papers \
        --output-dir ./structure_output
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams

# Import from parent module
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_utils import (
    setup_llm_api_keys,
    get_llm_class,
    get_token_limits,
    get_llm_params,
    get_tracing_config,
)
from tracing import setup_tracing, workflow_span

# Import local utils
from .extract_structure_utils import (
    Paper,
    Section,
    process_paper,
    extract_table_of_contents,
)


_MODULE_DIR = Path(__file__).parent
_CONFIG_PATH = str(_MODULE_DIR.parent / "mcp_agent.config.yaml")
_PROMPT_PATH = _MODULE_DIR / "section_analysis_prompt.txt"

app = MCPApp(name="structure_extractor", settings=_CONFIG_PATH)


def load_prompt() -> str:
    """Load the section analysis prompt."""
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


@dataclass
class SectionTask:
    """Represents a section to be analyzed."""
    paper_name: str
    section_title: str
    section_content: str
    table_of_contents: str
    output_path: Path


def collect_all_sections(sections: List[Section]) -> List[Tuple[str, str]]:
    """Recursively collect all sections and their content.

    Each section (including subsections) becomes a separate entry.
    Sections with no content are included with empty string.
    """
    result = []
    for section in sections:
        # Add every section, even if content is empty
        result.append((section.title, section.content.strip()))

        # Recursively add subsections
        if section.subsections:
            result.extend(collect_all_sections(section.subsections))

    return result


def prepare_section_tasks(
    paper: Paper,
    output_dir: Path
) -> List[SectionTask]:
    """Prepare section tasks for a paper."""
    toc = extract_table_of_contents(paper)
    paper_output_dir = output_dir / paper.name
    paper_output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    sections = collect_all_sections(paper.sections)

    for section_title, section_content in sections:
        # Create safe filename from section title
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in section_title)
        safe_title = safe_title.strip()[:50]  # Limit length
        output_path = paper_output_dir / f"{safe_title}.json"

        tasks.append(SectionTask(
            paper_name=paper.name,
            section_title=section_title,
            section_content=section_content,
            table_of_contents=toc,
            output_path=output_path,
        ))

    return tasks


async def analyze_section(
    task: SectionTask,
    prompt_template: str,
    params: RequestParams,
) -> Tuple[str, bool, str]:
    """Analyze a single section using LLM.

    Returns:
        Tuple of (section_title, success, result_or_error)
    """
    try:
        agent = Agent(
            name="section_analyzer",
            instruction=prompt_template,
        )

        llm_class = get_llm_class()
        llm = llm_class(agent=agent)

        message = f"""Analyze the following section from the paper "{task.paper_name}".

=== TABLE OF CONTENTS (for context) ===
{task.table_of_contents}

=== SECTION TO ANALYZE ===
Title: {task.section_title}

Content:
{task.section_content}
"""

        result = await llm.generate_str(message=message, request_params=params)

        # Save result to file
        task.output_path.write_text(result, encoding="utf-8")

        return (task.section_title, True, result)

    except Exception as e:
        error_msg = str(e)
        # Save error to file
        error_data = {"error": error_msg, "section_title": task.section_title}
        task.output_path.write_text(json.dumps(error_data, indent=2), encoding="utf-8")
        return (task.section_title, False, error_msg)


async def process_paper_sections(
    paper_dir: Path,
    output_dir: Path,
    prompt_template: str,
    params: RequestParams,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, List[Tuple[str, bool, str]]]:
    """Process all sections of a single paper.

    Returns:
        Tuple of (paper_name, list of (section_title, success, result))
    """
    try:
        paper = process_paper(paper_dir)
        print(f"  Parsed {paper.name}: {len(paper.sections)} top-level sections")

        tasks = prepare_section_tasks(paper, output_dir)
        print(f"  Prepared {len(tasks)} section tasks for {paper.name}")

        # Save table of contents
        toc_path = output_dir / paper.name / "_table_of_contents.txt"
        toc_path.write_text(extract_table_of_contents(paper), encoding="utf-8")

        results = []
        for task in tasks:
            async with semaphore:
                result = await analyze_section(task, prompt_template, params)
                results.append(result)
                status = "OK" if result[1] else "FAILED"
                print(f"    [{status}] {task.paper_name} / {task.section_title}")

        return (paper.name, results)

    except Exception as e:
        print(f"  Error processing {paper_dir.name}: {e}")
        return (paper_dir.name, [("_error", False, str(e))])


async def run_extraction(
    papers_dir: Path,
    output_dir: Path,
    max_concurrent: int = 5,
):
    """Run the structure extraction workflow for all papers."""
    setup_llm_api_keys()

    tracing_enabled, tracing_project, tracing_endpoint = get_tracing_config()
    setup_tracing(
        enabled=tracing_enabled,
        project_name=tracing_project,
        endpoint=tracing_endpoint,
    )

    prompt_template = load_prompt()
    base_max_tokens, _ = get_token_limits()
    temperature, max_iterations = get_llm_params()

    params = RequestParams(
        maxTokens=base_max_tokens,
        temperature=temperature,
        max_iterations=max_iterations,
    )

    # Get all paper directories
    paper_dirs = [d for d in papers_dir.iterdir() if d.is_dir()]
    print(f"Found {len(paper_dirs)} papers to process")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Semaphore to limit concurrent LLM calls
    semaphore = asyncio.Semaphore(max_concurrent)

    async with app.run():
        with workflow_span("structure-extraction", papers_dir=str(papers_dir)):
            # Process all papers concurrently
            tasks = [
                process_paper_sections(
                    paper_dir, output_dir, prompt_template, params, semaphore
                )
                for paper_dir in paper_dirs
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Summary
            print("\n" + "=" * 80)
            print("EXTRACTION SUMMARY")
            print("=" * 80)

            total_sections = 0
            successful_sections = 0

            for result in results:
                if isinstance(result, Exception):
                    print(f"  ERROR: {result}")
                    continue

                paper_name, section_results = result
                paper_total = len(section_results)
                paper_success = sum(1 for _, success, _ in section_results if success)
                total_sections += paper_total
                successful_sections += paper_success

                status = "OK" if paper_success == paper_total else f"{paper_success}/{paper_total}"
                print(f"  [{status}] {paper_name}")

            print("=" * 80)
            print(f"Total: {successful_sections}/{total_sections} sections analyzed successfully")
            print(f"Output saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract and analyze paper structure using LLM"
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        required=True,
        help="Directory containing paper folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save extraction results",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent LLM calls (default: 5)",
    )
    args = parser.parse_args()

    if not args.papers_dir.exists():
        print(f"Error: Papers directory not found: {args.papers_dir}")
        sys.exit(1)

    asyncio.run(run_extraction(
        papers_dir=args.papers_dir,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
    ))


if __name__ == "__main__":
    main()
