#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path
from opentelemetry import trace

# Setup PYTHONPATH for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from plan_generator.workflow import run_workflow

async def main():
    paper_dir = str(project_root / "2")
    output_file = str(project_root / "analysis_result.yaml")

    print("=" * 80)
    print("Plan Generator - Code Analysis")
    print("=" * 80)
    print(f"Paper directory: {paper_dir}")
    print(f"Output file: {output_file}")
    print("=" * 80)

    await run_workflow(paper_dir, output_file)

if __name__ == "__main__":
    asyncio.run(main())
