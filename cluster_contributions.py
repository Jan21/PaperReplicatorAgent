#!/home/jan/miniconda3/envs/vllm/bin/python
"""
Cluster ML paper contributions by contribution type using LLM semantic grouping.

Usage:
    python cluster_contributions.py [--input-dir DIR] [--output-dir DIR]
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams

# Add plan_generator to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from plan_generator.llm_utils import (
    setup_llm_api_keys,
    get_llm_class,
    get_token_limits,
    get_llm_params,
)

DEFAULT_INPUT_DIR = "outputs/contribution_extractor/deepseek_deepseek-v3.2"
DEFAULT_OUTPUT_DIR = "outputs/contribution_clustering"

_MODULE_DIR = Path(__file__).parent
_CONFIG_PATH = str(_MODULE_DIR / "plan_generator" / "mcp_agent.config.yaml")

app = MCPApp(name="contribution_clusterer", settings=_CONFIG_PATH)


def extract_paper_title(filename: str) -> str:
    """Extract paper title from filename like '2005.13406_Neural_heuristics_for_SAT_solving.md'."""
    # Remove extension
    name = Path(filename).stem
    # Remove arxiv ID prefix (pattern: YYMM.NNNNN_)
    name = re.sub(r"^\d{4}\.\d{4,5}_", "", name)
    # Replace underscores with spaces
    return name.replace("_", " ")


def parse_json_from_markdown(content: str) -> list:
    """Parse JSON from markdown content, handling code fences."""
    # Try to find JSON in code fences
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Try parsing the whole content as JSON
        json_str = content.strip()

    return json.loads(json_str)


def extract_all_contributions(input_dir: Path) -> list:
    """Extract all contributions from markdown files in input directory."""
    all_contributions = []

    md_files = sorted(input_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        paper_title = extract_paper_title(md_file.name)

        try:
            contributions = parse_json_from_markdown(content)
            for idx, contrib in enumerate(contributions):
                contrib["paper_name"] = paper_title
                contrib["source_file"] = md_file.name
                contrib["contribution_index"] = idx
                all_contributions.append(contrib)
            print(f"  {md_file.name}: {len(contributions)} contributions")
        except json.JSONDecodeError as e:
            print(f"  ERROR parsing {md_file.name}: {e}")

    return all_contributions


def get_unique_contribution_types(contributions: list) -> list:
    """Get unique contribution types from all contributions."""
    types = set()
    for contrib in contributions:
        if "contribution_type" in contrib:
            types.add(contrib["contribution_type"])
    return sorted(types)


def build_clustering_prompt(contribution_types: list, user_feedback: list) -> str:
    """Build the prompt for LLM clustering."""
    types_list = "\n".join(f"- {t}" for t in contribution_types)

    prompt = f"""You are tasked with clustering contribution types from machine learning research papers.

Here are all the unique contribution types extracted from papers:

{types_list}

Please group these into semantic clusters. Many phrases refer to the same concept with slight variations (e.g., case differences, similar wording).

For each cluster:
1. Create a clear, canonical name that describes the cluster
2. List all the original phrases that belong to this cluster

Return your response as a JSON array where each element has:
- "canonical_name": A clear, descriptive name for the cluster
- "member_phrases": An array of all original phrases that belong to this cluster

IMPORTANT: Every phrase from the input list must appear in exactly one cluster.

Example output format:
```json
[
    {{
        "canonical_name": "Neural Architecture Design",
        "member_phrases": ["Novel Architecture", "Architecture Design", "Novel Architecture / Oracle Factory"]
    }},
    {{
        "canonical_name": "Training & Optimization",
        "member_phrases": ["Training/Optimization Procedure", "Novel Training Paradigm"]
    }}
]
```
"""

    if user_feedback:
        feedback_text = "\n\n".join(f"Feedback {i+1}: {fb}" for i, fb in enumerate(user_feedback))
        prompt += f"\n\nADDITIONAL INSTRUCTIONS FROM USER:\n{feedback_text}"

    return prompt


def parse_clustering_response(response: str) -> list:
    """Parse the LLM clustering response into a list of clusters."""
    # Try to find JSON in the response
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Try to find raw JSON array
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            raise ValueError("Could not find JSON in response")

    return json.loads(json_str)


def print_clusters(clusters: list):
    """Print clusters in a readable format."""
    print("\n" + "=" * 60)
    print("CLUSTERING RESULT")
    print("=" * 60)

    for i, cluster in enumerate(clusters, 1):
        canonical = cluster.get("canonical_name", "Unknown")
        members = cluster.get("member_phrases", [])
        print(f"\n=== Cluster {i}: {canonical} ({len(members)} phrases) ===")
        for member in sorted(members):
            print(f"  - {member}")

    print("\n" + "=" * 60)
    print(f"Total clusters: {len(clusters)}")
    total_phrases = sum(len(c.get("member_phrases", [])) for c in clusters)
    print(f"Total phrases: {total_phrases}")
    print("=" * 60)


async def run_clustering_loop(contribution_types: list) -> list:
    """Run the interactive clustering loop with LLM."""
    setup_llm_api_keys()

    base_max_tokens, _ = get_token_limits()
    temperature, max_iterations = get_llm_params()

    params = RequestParams(
        maxTokens=base_max_tokens,
        temperature=temperature,
        max_iterations=max_iterations,
    )

    user_feedback = []

    async with app.run():
        agent = Agent(
            name="clustering_agent",
            instruction="You are a helpful assistant that clusters contribution types from ML papers.",
        )

        llm_class = get_llm_class()
        llm = llm_class(agent=agent)

        while True:
            print("\n" + "-" * 40)
            print("Sending clustering request to LLM...")
            print("-" * 40)

            prompt = build_clustering_prompt(contribution_types, user_feedback)
            response = await llm.generate_str(message=prompt, request_params=params)

            try:
                clusters = parse_clustering_response(response)
                print_clusters(clusters)

                # Check coverage
                all_members = set()
                for cluster in clusters:
                    all_members.update(cluster.get("member_phrases", []))

                missing = set(contribution_types) - all_members
                extra = all_members - set(contribution_types)

                if missing:
                    print(f"\nWARNING: Missing phrases: {missing}")
                if extra:
                    print(f"\nWARNING: Extra phrases not in original: {extra}")

            except (json.JSONDecodeError, ValueError) as e:
                print(f"\nERROR parsing LLM response: {e}")
                print("Raw response:")
                print(response[:2000])
                user_input = input("\nPress Enter to retry, or type feedback: ").strip()
                if user_input:
                    user_feedback.append(user_input)
                continue

            print("\n")
            user_input = input("Accept clustering? [y/yes to accept, or enter feedback to refine]: ").strip()

            if user_input.lower() in ("y", "yes", ""):
                print("\nClustering accepted!")
                return clusters
            else:
                user_feedback.append(user_input)
                print(f"\nAdded feedback. Total feedback items: {len(user_feedback)}")


def generate_final_outputs(
    all_contributions: list,
    clusters: list,
    output_dir: Path
):
    """Generate the final clustered output files."""
    # Build reverse lookup: phrase -> canonical_name
    phrase_to_cluster = {}
    for cluster in clusters:
        canonical = cluster["canonical_name"]
        for phrase in cluster["member_phrases"]:
            phrase_to_cluster[phrase] = canonical

    # Generate flat output
    flat_output = []
    for contrib in all_contributions:
        contrib_copy = contrib.copy()
        original_type = contrib_copy.get("contribution_type", "")
        contrib_copy["cluster_name"] = phrase_to_cluster.get(original_type, "Unknown")
        flat_output.append(contrib_copy)

    # Generate nested output
    nested_output = {}
    for cluster in clusters:
        canonical = cluster["canonical_name"]
        nested_output[canonical] = {
            "canonical_name": canonical,
            "member_phrases": cluster["member_phrases"],
            "contributions": []
        }

    # Add "Unknown" cluster for any unmatched
    nested_output["Unknown"] = {
        "canonical_name": "Unknown",
        "member_phrases": [],
        "contributions": []
    }

    for contrib in flat_output:
        cluster_name = contrib["cluster_name"]
        nested_output[cluster_name]["contributions"].append(contrib)

    # Remove empty "Unknown" cluster if not used
    if not nested_output["Unknown"]["contributions"]:
        del nested_output["Unknown"]

    # Save files
    flat_path = output_dir / "clustered_flat.json"
    with open(flat_path, "w", encoding="utf-8") as f:
        json.dump(flat_output, f, indent=2, ensure_ascii=False)
    print(f"Saved flat output to {flat_path}")

    nested_path = output_dir / "clustered_nested.json"
    with open(nested_path, "w", encoding="utf-8") as f:
        json.dump(nested_output, f, indent=2, ensure_ascii=False)
    print(f"Saved nested output to {nested_path}")

    return flat_output, nested_output


async def main():
    parser = argparse.ArgumentParser(description="Cluster ML paper contributions by type")
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory with contribution markdown files (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for clustered results (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract all contributions
    print("\n" + "=" * 60)
    print("STEP 1: Extracting contributions from markdown files")
    print("=" * 60)
    all_contributions = extract_all_contributions(input_dir)
    print(f"\nTotal contributions extracted: {len(all_contributions)}")

    # Save intermediate file
    all_contrib_path = output_dir / "all_contributions.json"
    with open(all_contrib_path, "w", encoding="utf-8") as f:
        json.dump(all_contributions, f, indent=2, ensure_ascii=False)
    print(f"Saved to {all_contrib_path}")

    # Step 2: Get unique contribution types
    contribution_types = get_unique_contribution_types(all_contributions)
    print(f"\nUnique contribution types: {len(contribution_types)}")

    # Step 3: Interactive LLM clustering
    print("\n" + "=" * 60)
    print("STEP 2: Interactive LLM clustering")
    print("=" * 60)
    clusters = await run_clustering_loop(contribution_types)

    # Save cluster mapping
    cluster_mapping_path = output_dir / "cluster_mapping.json"
    with open(cluster_mapping_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    print(f"Saved cluster mapping to {cluster_mapping_path}")

    # Step 4: Generate final outputs
    print("\n" + "=" * 60)
    print("STEP 3: Generating final clustered outputs")
    print("=" * 60)
    generate_final_outputs(all_contributions, clusters, output_dir)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
