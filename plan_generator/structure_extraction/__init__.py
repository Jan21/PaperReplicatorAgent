"""Structure extraction module for paper analysis."""

from .extract_structure_utils import (
    Section,
    Paper,
    MarkdownParser,
    process_paper,
    process_all_papers,
    concatenate_paper_pages,
    extract_table_of_contents,
    get_sections_content,
    get_section_descriptions,
)

__all__ = [
    "Section",
    "Paper",
    "MarkdownParser",
    "process_paper",
    "process_all_papers",
    "concatenate_paper_pages",
    "extract_table_of_contents",
    "get_sections_content",
    "get_section_descriptions",
]
