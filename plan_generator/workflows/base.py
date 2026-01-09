"""Shared utilities for workflow modules."""

import os
import re
from pathlib import Path

import yaml


def extract_page_number(filename: str) -> int:
    """Extract page number from filename like 'paper_page_1.txt' -> 1"""
    match = re.search(r'_page_(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    return -1


def load_paper_content(paper_dir: str) -> str | None:
    """
    Load paper content from a directory containing page files.

    Args:
        paper_dir: Directory path containing *_page_N.txt files

    Returns:
        Concatenated paper content or None if no files found
    """
    txt_files = []
    for filename in os.listdir(paper_dir):
        if filename.endswith(".txt") and "_page_" in filename:
            page_num = extract_page_number(filename)
            if page_num > 0:
                txt_files.append((page_num, filename))

    if not txt_files:
        return None

    # Sort by page number
    txt_files.sort(key=lambda x: x[0])

    # Concatenate all pages in order
    paper_parts = []
    for page_num, filename in txt_files:
        file_path = os.path.join(paper_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            paper_parts.append(f.read())

    content = "\n\n\n".join(paper_parts)
    print(f"Loaded {len(txt_files)} page(s) from {paper_dir} ({len(content)} chars)")
    return content


def get_workflow_config() -> str:
    """
    Get the configured workflow type from config file.

    Returns:
        Workflow identifier string (default: 'plan_generator')
    """
    config_path = Path(__file__).parent.parent / "mcp_agent.config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config.get("workflow", "plan_generator")
