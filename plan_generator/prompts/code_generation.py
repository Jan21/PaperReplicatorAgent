"""Code generation prompt for producing actual implementation files from a plan."""

INTRO = """You are an expert software engineer tasked with implementing code based on a detailed reproduction plan.
Your goal is to generate complete, working, well-documented code files."""

CONTEXT_SECTION = """# CONTEXT
You have been provided with:
1. **Implementation Plan**: A comprehensive YAML plan specifying file structure, components, algorithms, and implementation details
2. **File Group**: The specific group of files you should generate in this batch

Follow the plan precisely. Implement all algorithms, formulas, and components as specified."""

GENERATION_RULES = """# GENERATION RULES

## Code Quality Requirements:
- Write clean, readable, production-quality code
- Include comprehensive docstrings for all classes and functions
- Add inline comments for complex logic or algorithms
- Follow PEP 8 style guidelines for Python
- Use type hints where appropriate
- Handle edge cases and errors gracefully

## Implementation Requirements:
- Implement ALL algorithms and formulas exactly as specified in the plan
- Use the exact file names and structure from the plan
- Include all hyperparameters with their specified values
- Create proper imports between modules
- Make code executable and testable

## What NOT to do:
- Do not skip or simplify algorithms
- Do not use placeholder comments like "# TODO: implement this"
- Do not leave functions empty or partially implemented
- Do not deviate from the specified file structure"""

OUTPUT_FORMAT = """# OUTPUT FORMAT

For EACH file you generate, use this exact format:

```
===FILE: <relative_path/filename.py>===
<complete file contents here>
===END_FILE===
```

Example:
```
===FILE: models/neural_net.py===
\"\"\"Neural network implementation.\"\"\"

import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # ... rest of implementation
===END_FILE===

===FILE: utils/helpers.py===
\"\"\"Helper utilities.\"\"\"

def compute_metric(pred, target):
    # ... implementation
===END_FILE===
```

IMPORTANT:
- Generate ALL files in the requested group
- Each file must be complete and functional
- Use the exact paths from the implementation plan"""

BATCH_INSTRUCTIONS = {
    "core": """# CURRENT TASK: Generate CORE Implementation Files

Generate the core algorithm and model files. These are the primary components that implement
the paper's main contributions. Focus on:
- Main algorithm implementations
- Core model/architecture classes
- Primary data structures
- Mathematical formulations and computations

These files should be self-contained where possible, with clear interfaces for other modules.""",

    "support": """# CURRENT TASK: Generate SUPPORT Files

Generate supporting modules and utilities. These help the core implementation function. Focus on:
- Utility functions and helpers
- Data loading and preprocessing
- Configuration handling
- Common operations used across the codebase

These files should integrate well with the core files generated previously.""",

    "experiment": """# CURRENT TASK: Generate EXPERIMENT and Evaluation Files

Generate experiment scripts, evaluation code, and entry points. Focus on:
- Training/execution scripts (main.py, train.py, run.py, etc.)
- Evaluation and metrics computation
- Experiment configuration
- Result visualization and logging

These files should tie everything together into runnable experiments.""",

    "docs": """# CURRENT TASK: Generate DOCUMENTATION Files

Generate documentation and setup files. Focus on:
- README.md with clear usage instructions
- requirements.txt with all dependencies
- Any additional documentation needed

These files should make the project easy to understand and use."""
}

def get_code_generation_prompt(batch_type: str) -> str:
    """
    Get the code generation prompt for a specific batch type.

    Args:
        batch_type: One of 'core', 'support', 'experiment', 'docs'

    Returns:
        Complete prompt string for code generation
    """
    batch_instruction = BATCH_INSTRUCTIONS.get(batch_type, BATCH_INSTRUCTIONS["core"])

    return "\n\n".join([
        INTRO,
        CONTEXT_SECTION,
        batch_instruction,
        GENERATION_RULES,
        OUTPUT_FORMAT,
    ])

# Default prompt for single-batch generation
CODE_GENERATION_PROMPT = get_code_generation_prompt("core")
