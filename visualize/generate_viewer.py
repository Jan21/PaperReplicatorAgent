#!/usr/bin/env python3
"""
Generate the experiments viewer HTML file with embedded data from YAML files.
"""

import os
import re
import json
import yaml
from pathlib import Path


def extract_yaml_from_markdown(content: str) -> str:
    """Extract YAML content from markdown code blocks."""
    # Remove ```yaml and ``` wrappers
    match = re.search(r'```yaml\s*([\s\S]*?)```', content)
    if match:
        return match.group(1)
    return content


def parse_experiment_file(filepath: Path) -> dict:
    """Parse a single experiment markdown file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    yaml_content = extract_yaml_from_markdown(content)

    try:
        data = yaml.safe_load(yaml_content)
        if data and 'complete_experiment_summary' in data:
            return data['complete_experiment_summary']
        return data
    except yaml.YAMLError as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def process_experiments(experiments: dict) -> dict:
    """Process experiments to ensure consistent structure."""
    processed = {}
    for key, exp in experiments.items():
        if not isinstance(exp, dict):
            continue
        processed[key] = {
            'name': exp.get('name', ''),
            'paper_section': exp.get('paper_section', ''),
            'experiment_type': exp.get('experiment_type', 'other'),
            'purpose': exp.get('purpose', ''),
            'benchmarks': exp.get('benchmarks', []),
            'baselines': exp.get('baselines', []),
            'conclusion': exp.get('conclusion', {}),
            'detailed_description': exp.get('detailed_description', '')
        }
    return processed


def load_all_papers(input_dir: Path) -> list:
    """Load all paper experiment files from the directory."""
    papers = []

    for filepath in sorted(input_dir.glob('*.md')):
        print(f"Processing: {filepath.name}")
        data = parse_experiment_file(filepath)

        if data:
            paper = {
                'filename': filepath.name,
                'paper_title': data.get('paper_title', filepath.stem),
                'paper_focus': data.get('paper_focus', ''),
                'experiment_inventory': data.get('experiment_inventory', {
                    'total_experiments': 0,
                    'main_experiments': 0,
                    'ablation_studies': 0,
                    'other_experiments': 0
                }),
                'experiments': process_experiments(data.get('experiments', {})),
                'summary_table': data.get('summary_table', []),
                'cross_experiment_analysis': data.get('cross_experiment_analysis', {}),
                'overall_conclusions': data.get('overall_conclusions', {})
            }

            # Ensure inventory values are integers
            inv = paper['experiment_inventory']
            for key in ['total_experiments', 'main_experiments', 'ablation_studies', 'other_experiments']:
                if key in inv:
                    try:
                        inv[key] = int(inv[key]) if inv[key] else 0
                    except (ValueError, TypeError):
                        inv[key] = 0

            papers.append(paper)

    return papers


def generate_html(papers: list, template_path: Path, output_path: Path):
    """Generate the final HTML file with embedded data."""
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # Convert papers to JSON
    papers_json = json.dumps(papers, indent=2, ensure_ascii=False)

    # Replace placeholder with actual data
    html = template.replace('PAPERS_DATA_PLACEHOLDER', papers_json)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nGenerated: {output_path}")
    print(f"Total papers: {len(papers)}")
    total_experiments = sum(p['experiment_inventory'].get('total_experiments', 0) for p in papers)
    print(f"Total experiments: {total_experiments}")


def main():
    # Determine paths
    script_dir = Path(__file__).parent
    input_dir = script_dir.parent / 'outputs' / 'experiment_extractor'
    template_path = script_dir / 'experiments_viewer.html'
    output_path = script_dir / 'index.html'

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    if not template_path.exists():
        print(f"Error: Template file not found: {template_path}")
        return

    print(f"Loading papers from: {input_dir}")
    papers = load_all_papers(input_dir)

    if not papers:
        print("No papers found!")
        return

    generate_html(papers, template_path, output_path)


if __name__ == '__main__':
    main()
