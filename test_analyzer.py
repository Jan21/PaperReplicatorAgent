#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# Setup PYTHONPATH for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from plan_generator.workflow import run_workflow_batch
from plan_generator.workflows.base import get_workflow_config
from plan_generator.llm_utils import get_model_name


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in file paths."""
    # Replace slashes and other problematic characters
    return model.replace("/", "_").replace("\\", "_").replace(":", "_")


async def main():
    papers_dir = project_root / "papers"
    workflow_name = get_workflow_config()
    model_name = sanitize_model_name(get_model_name())

    # Create output directory
    output_dir = project_root / "outputs" / workflow_name / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Paper Analyzer - Running workflow: {workflow_name}")
    print("=" * 80)
    print(f"Papers directory: {papers_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Get all paper directories (skip hidden directories)
    paper_dirs = sorted([
        d for d in papers_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    print(f"Found {len(paper_dirs)} paper(s) to process\n")

    # Build list of papers to process (skip existing outputs)
    papers_to_process = []
    skipped = []

    for paper_dir in paper_dirs:
        output_file = output_dir / f"{paper_dir.name}.md"
        if output_file.exists():
            print(f"Skipping {paper_dir.name} - output already exists")
            skipped.append(paper_dir.name)
        else:
            papers_to_process.append((str(paper_dir), str(output_file)))

    results = {"success": list(skipped), "failed": []}

    if papers_to_process:
        print(f"\nProcessing {len(papers_to_process)} paper(s) concurrently...\n")
        batch_results = await run_workflow_batch(papers_to_process)

        for paper_dir, success, _ in batch_results:
            paper_name = Path(paper_dir).name
            if success:
                results["success"].append(paper_name)
            else:
                results["failed"].append(paper_name)

    # Print summary
    print("=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total papers: {len(paper_dirs)}")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")

    if results["failed"]:
        print("\nFailed papers:")
        for name in results["failed"]:
            print(f"  - {name}")


if __name__ == "__main__":
    asyncio.run(main())


