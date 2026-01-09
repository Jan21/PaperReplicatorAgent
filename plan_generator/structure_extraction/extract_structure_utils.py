#!/usr/bin/env python3
"""
Script to concatenate markdown pages from papers and parse into hierarchical structure.
"""

import argparse
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Section:
    """Represents a section in the document with its hierarchy."""
    level: int
    title: str
    content: str = ""
    subsections: List['Section'] = field(default_factory=list)

    def to_dict(self):
        """Convert section to dictionary representation."""
        return {
            'level': self.level,
            'title': self.title,
            'content': self.content,
            'subsections': [sub.to_dict() for sub in self.subsections]
        }

    def __repr__(self, indent=0):
        """Pretty print the section hierarchy."""
        result = "  " * indent + f"{'#' * self.level} {self.title}\n"
        if self.content.strip():
            content_preview = self.content.strip()[:100].replace('\n', ' ')
            result += "  " * indent + f"  Content: {content_preview}...\n"
        for subsection in self.subsections:
            result += subsection.__repr__(indent + 1)
        return result


@dataclass
class Paper:
    """Represents a parsed paper with hierarchical structure."""
    name: str
    full_text: str
    sections: List[Section] = field(default_factory=list)

    def to_dict(self):
        """Convert paper to dictionary representation."""
        return {
            'name': self.name,
            'full_text': self.full_text,
            'sections': [section.to_dict() for section in self.sections]
        }


class MarkdownParser:
    """Parser to convert markdown text into hierarchical structure."""

    def __init__(self):
        self.sections = []
        self.current_content = []

    def _detect_heading(self, line: str) -> Optional[tuple]:
        """
        Detect if a line is a heading (markdown or numbered section).

        Returns:
            tuple of (level, title) if heading detected, None otherwise
        """
        # Pattern 1: Markdown headings (# Title, ## Title, etc.)
        md_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if md_match:
            level = len(md_match.group(1))
            title = md_match.group(2).strip()
            return (level, title)

        # Pattern 2: Numbered sections (e.g., "5 CONCLUSIONS", "3.1 Methods", "A.2 Appendix")
        # Also handles OCR issues where heading merges with content (e.g., "5 CONCLUSIONSIn this work...")
        # Matches: number/letter prefix + uppercase words only (no lowercase allowed in title)
        numbered_match = re.match(
            r'^([A-Z]?\d+(?:\.\d+)?)\s+([A-Z][A-Z0-9 \-&]*)',
            line.strip()
        )
        if numbered_match:
            section_num = numbered_match.group(1)
            title_part = numbered_match.group(2).strip()

            # Fix OCR merge issue: trim trailing uppercase letter(s) if followed by lowercase
            # e.g., "CONCLUSIONS AND FUTURE WORKI" -> "CONCLUSIONS AND FUTURE WORK"
            # Look for pattern like "WORKIn" where uppercase title merges with sentence start
            remaining = line.strip()[numbered_match.end():]
            if remaining and remaining[0].islower() and title_part and title_part[-1].isupper():
                # Pattern detected: uppercase char followed by lowercase (e.g., "In")
                # This suggests the last char of title belongs to the next word
                # Only trim if it looks like a word start (single uppercase + lowercase)
                title_part = title_part[:-1]

            title_part = title_part.strip()
            # Only accept if title is at least 3 chars (avoid false positives)
            if len(title_part) >= 3:
                title = f"{section_num} {title_part}"
                # Determine level based on numbering depth
                if '.' in section_num:
                    level = 3  # Subsection like 3.1
                else:
                    level = 2  # Main section like 5
                return (level, title)

        return None

    def _is_duplicate_section(self, title: str, section_stack: List[Section]) -> bool:
        """Check if a section with this title already exists at the same level."""
        # Check in parent's subsections or root sections
        if section_stack:
            parent = section_stack[-1]
            return any(s.title == title for s in parent.subsections)
        else:
            return any(s.title == title for s in self.sections)

    def parse(self, text: str) -> List[Section]:
        """Parse markdown text and return list of top-level sections."""
        lines = text.split('\n')
        section_stack = []  # Stack to track current hierarchy
        current_content = []

        for line in lines:
            # Check if line is a heading (markdown or numbered)
            heading_info = self._detect_heading(line)

            if heading_info:
                level, title = heading_info

                # Prepare stack for new section (pop sections at same or deeper level)
                temp_stack = section_stack[:]
                while temp_stack and temp_stack[-1].level >= level:
                    temp_stack.pop()

                # Skip duplicate sections (OCR artifacts)
                if self._is_duplicate_section(title, temp_stack):
                    current_content.append(line)
                    continue

                # Save content to the last section before starting new one
                if section_stack:
                    section_stack[-1].content = '\n'.join(current_content).strip()
                    current_content = []

                new_section = Section(level=level, title=title)

                # Pop sections from stack that are at same or deeper level
                while section_stack and section_stack[-1].level >= level:
                    section_stack.pop()

                # Add new section to parent or to root
                if section_stack:
                    section_stack[-1].subsections.append(new_section)
                else:
                    self.sections.append(new_section)

                section_stack.append(new_section)
            else:
                # Accumulate content for current section
                current_content.append(line)

        # Save final content
        if section_stack:
            section_stack[-1].content = '\n'.join(current_content).strip()

        return self.sections


def concatenate_paper_pages(paper_dir: Path) -> str:
    """
    Concatenate all .txt files from a paper directory in order.

    Args:
        paper_dir: Path to the paper directory

    Returns:
        Concatenated text from all pages
    """
    # Get all .txt files
    txt_files = sorted([f for f in paper_dir.glob("*.txt")])

    # Extract page numbers and sort
    def get_page_number(filename):
        match = re.search(r'page_(\d+)\.txt$', filename.name)
        return int(match.group(1)) if match else 0

    txt_files.sort(key=get_page_number)

    # Concatenate all pages
    full_text = []
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            full_text.append(content)
            full_text.append('\n\n')  # Add spacing between pages

    return ''.join(full_text)


def process_paper(paper_dir: Path) -> Paper:
    """
    Process a single paper: concatenate pages and parse structure.

    Args:
        paper_dir: Path to the paper directory

    Returns:
        Paper object with hierarchical structure
    """
    paper_name = paper_dir.name

    # Concatenate all pages
    full_text = concatenate_paper_pages(paper_dir)

    # Parse markdown structure
    parser = MarkdownParser()
    sections = parser.parse(full_text)

    return Paper(name=paper_name, full_text=full_text, sections=sections)


def process_all_papers(markdown_dir: Path) -> List[Paper]:
    """
    Process all papers in the markdown directory.

    Args:
        markdown_dir: Path to the directory containing paper folders

    Returns:
        List of Paper objects
    """
    papers = []

    # Get all subdirectories (paper folders)
    paper_dirs = [d for d in markdown_dir.iterdir() if d.is_dir()]

    for paper_dir in sorted(paper_dirs):
        try:
            paper = process_paper(paper_dir)
            papers.append(paper)
            print(f"✓ Processed: {paper.name}")
        except Exception as e:
            print(f"✗ Error processing {paper_dir.name}: {e}")

    return papers


def extract_table_of_contents(paper: Paper, indent: str = "  ") -> str:
    """
    Extract hierarchical table of contents from a paper.

    Args:
        paper: Paper object with parsed sections
        indent: String to use for indentation (default: 2 spaces)

    Returns:
        String representation of section hierarchy (like a file tree)
    """
    def format_section(section: Section, level: int = 0) -> str:
        lines = []
        prefix = indent * level
        lines.append(f"{prefix}{section.title}")
        for subsection in section.subsections:
            lines.append(format_section(subsection, level + 1))
        return '\n'.join(lines)

    toc_lines = []
    for section in paper.sections:
        toc_lines.append(format_section(section))

    return '\n'.join(toc_lines)


def get_sections_content(paper: Paper, section_names: List[str]) -> str:
    """
    Get concatenated content of specified sections.

    Args:
        paper: Paper object with parsed sections
        section_names: List of section titles to extract content from

    Returns:
        Concatenated string of the text from the specified sections
    """
    def find_sections(sections: List[Section], names: List[str]) -> List[Section]:
        """Recursively find all sections matching the given names."""
        found = []
        for section in sections:
            if section.title in names:
                found.append(section)
            found.extend(find_sections(section.subsections, names))
        return found

    def get_full_content(section: Section) -> str:
        """Get content of a section including all its subsections."""
        content_parts = [section.content] if section.content.strip() else []
        for subsection in section.subsections:
            content_parts.append(get_full_content(subsection))
        return '\n\n'.join(part for part in content_parts if part.strip())

    matching_sections = find_sections(paper.sections, section_names)

    contents = []
    for section in matching_sections:
        contents.append(get_full_content(section))

    return '\n\n'.join(contents)


def get_section_descriptions(paper: Paper, max_length: int = 200) -> dict:
    """
    Get short text descriptions of each section.

    Args:
        paper: Paper object with parsed sections
        max_length: Maximum length of each description (default: 200 chars)

    Returns:
        Dictionary mapping section titles to their short descriptions
    """
    def get_description(section: Section) -> str:
        """Extract a short description from section content."""
        content = section.content.strip()
        if not content:
            # If no direct content, try to get from first subsection
            if section.subsections:
                content = section.subsections[0].content.strip()

        if not content:
            return ""

        # Clean up the content and truncate
        content = ' '.join(content.split())  # Normalize whitespace
        if len(content) > max_length:
            # Try to cut at a sentence or word boundary
            truncated = content[:max_length]
            last_period = truncated.rfind('.')
            last_space = truncated.rfind(' ')

            if last_period > max_length * 0.5:
                return truncated[:last_period + 1]
            elif last_space > 0:
                return truncated[:last_space] + "..."
            else:
                return truncated + "..."
        return content

    def process_sections(sections: List[Section], result: dict, prefix: str = "") -> None:
        """Recursively process sections and add to result dict."""
        for section in sections:
            key = f"{prefix}{section.title}" if prefix else section.title
            result[key] = get_description(section)
            process_sections(section.subsections, result, prefix=f"{key} > ")

    descriptions = {}
    process_sections(paper.sections, descriptions)
    return descriptions


def main():
    """Main function to demonstrate usage."""
    parser = argparse.ArgumentParser(
        description="Extract structure from a paper in markdown format."
    )
    parser.add_argument(
        "paper_dir",
        type=Path,
        help="Path to the paper directory containing markdown .txt files"
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        help="Section names to extract content from (for get_sections_content)"
    )
    args = parser.parse_args()

    paper_dir = args.paper_dir

    if not paper_dir.exists():
        print(f"Error: Directory '{paper_dir}' not found")
        return None

    if not paper_dir.is_dir():
        print(f"Error: '{paper_dir}' is not a directory")
        return None

    # Process the paper
    print(f"Processing paper: {paper_dir.name}")
    paper = process_paper(paper_dir)

    # 1. Show table of contents
    print(f"\n{'='*80}")
    print("TABLE OF CONTENTS")
    print(f"{'='*80}")
    toc = extract_table_of_contents(paper)
    print(toc)

    # 2. Show section descriptions
    print(f"\n{'='*80}")
    print("SECTION DESCRIPTIONS")
    print(f"{'='*80}")
    descriptions = get_section_descriptions(paper)
    for section_title, description in descriptions.items():
        print(f"\n[{section_title}]")
        if description:
            print(f"  {description}")
        else:
            print("  (no content)")

    # 3. Show content of specific sections if requested
    if args.sections:
        print(f"\n{'='*80}")
        print(f"CONTENT OF SECTIONS: {args.sections}")
        print(f"{'='*80}")
        content = get_sections_content(paper, args.sections)
        print(content if content else "(no matching sections found)")

    return paper


if __name__ == "__main__":
    main()
