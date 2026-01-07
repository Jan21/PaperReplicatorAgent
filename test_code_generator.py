#!/usr/bin/env python3
"""
Test script for the code generation workflow.

This script demonstrates the two-step process:
1. Analyze a paper to generate an implementation plan (or use existing plan)
2. Generate actual code files from the plan

Usage:
    # Using an existing plan file:
    python test_code_generator.py --plan analysis_result.yaml --project my_project

    # Full pipeline (analyze paper then generate code):
    python test_code_generator.py --paper papers/1909.11588_Graph_Neural_Reasoning_May_Fail_in_Certifying_Bool

    # Just generate code from a plan:
    python test_code_generator.py --plan-only analysis_result.yaml
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Setup PYTHONPATH for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from plan_generator.workflow import run_workflow
from plan_generator.code_generator import run_code_generation


async def main():
    parser = argparse.ArgumentParser(
        description="Paper analysis and code generation pipeline"
    )

    # Input options (mutually exclusive group)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--paper", "-d",
        help="Directory containing paper page files (*_page_N.txt). Will run full pipeline."
    )
    input_group.add_argument(
        "--plan-only", "-p",
        help="Path to existing plan file. Will only run code generation."
    )

    # Optional arguments
    parser.add_argument(
        "--plan", "-o",
        default="analysis_result.yaml",
        help="Output path for the analysis plan (default: analysis_result.yaml)"
    )
    parser.add_argument(
        "--project", "-n",
        help="Name for the generated project directory (default: derived from paper/plan name)"
    )
    parser.add_argument(
        "--skip-generation", "-s",
        action="store_true",
        help="Only run analysis, skip code generation"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Paper Replicator - Full Pipeline")
    print("=" * 80)

    # Determine project name
    if args.project:
        project_name = args.project
    elif args.paper:
        # Extract name from paper directory
        project_name = Path(args.paper).name
        # Truncate to reasonable length
        if len(project_name) > 50:
            project_name = project_name[:50]
    elif args.plan_only:
        project_name = Path(args.plan_only).stem.replace('analysis_', '').replace('_result', '')
    else:
        project_name = "generated_project"

    plan_file = args.plan

    # Step 1: Paper Analysis (if starting from paper)
    if args.paper:
        print(f"\nStep 1: Analyzing Paper")
        print(f"  Paper directory: {args.paper}")
        print(f"  Output plan: {plan_file}")
        print("-" * 40)

        await run_workflow(args.paper, plan_file)

        print(f"\nAnalysis complete. Plan saved to: {plan_file}")

        if args.skip_generation:
            print("\nSkipping code generation (--skip-generation flag set)")
            return

    else:
        # Using existing plan
        plan_file = args.plan_only
        print(f"\nUsing existing plan: {plan_file}")

    # Step 2: Code Generation
    print(f"\nStep 2: Generating Code")
    print(f"  Plan file: {plan_file}")
    print(f"  Project name: {project_name}")
    print("-" * 40)

    result = await run_code_generation(plan_file, project_name)

    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print(f"\nGenerated project location: projects/{project_name}/")
    print("\nNext steps:")
    print("  1. cd projects/{project_name}")
    print("  2. Review generated files")
    print("  3. pip install -r requirements.txt (if present)")
    print("  4. Run tests or experiments")


if __name__ == "__main__":
    asyncio.run(main())
