"""Code generation prompt for producing actual implementation files from a plan."""

INTRO = """You are an expert software engineer implementing a research paper reproduction.
Your code MUST be functional, runnable, and bug-free. The user will run your code directly."""

CRITICAL_REQUIREMENTS = """# CRITICAL REQUIREMENTS - READ CAREFULLY

## 1. NAMING CONSISTENCY (Most Common Bug Source)
Before writing ANY import or class instantiation:
- Check the EXACT class/function name you defined or will define
- Use that EXACT name - not a similar one, not a variant
- Example: If you define `class GraphNetwork`, import `GraphNetwork`, NOT `GraphNeuralNetwork`

## 2. IMPORT CORRECTNESS
- Only import what actually exists in the files
- Use relative imports within the package (from .module import X)
- Check that imported names match EXACTLY what's exported
- If you're unsure of a name from a previous batch, use a simple consistent naming scheme

## 3. RUNNABLE CODE
- Every script must be runnable: `python script.py`
- Include `if __name__ == "__main__":` blocks in executable files
- Provide default values and argument parsing for all parameters
- Handle missing files/data gracefully with clear error messages

## 4. NO PLACEHOLDERS OR STUBS
- Every function must have a complete implementation
- No `pass`, `...`, `# TODO`, or `raise NotImplementedError`
- No comments like "implement this" or "add logic here"
- If something is complex, implement a working simplified version

## 5. DEPENDENCY CHAIN
- Files in this batch may depend on files from previous batches
- Check the "Previously Generated Files" section for what exists
- Your imports from those files must use the correct names"""

VERIFICATION_CHECKLIST = """# BEFORE OUTPUTTING EACH FILE, VERIFY:

□ All imports reference real modules/classes with correct names
□ All class instantiations use the exact class name defined
□ All function calls use correct function names and signatures
□ All file paths in the code are relative and correct
□ Main scripts have working entry points
□ No placeholder code or TODOs remain
□ Error handling exists for file operations and user input"""

GENERATION_RULES = """# CODE GENERATION RULES

## File Structure:
- Use the exact file paths from the implementation plan
- Create proper __init__.py files to make packages importable
- Use relative imports within the package

## Code Style:
- Clear docstrings for all public classes and functions
- Type hints on function signatures
- Descriptive variable names
- Group related functionality together

## Error Handling:
- Validate inputs at function boundaries
- Provide helpful error messages
- Handle file not found, invalid input gracefully
- Use try/except around I/O operations

## Configuration:
- Use argparse for command-line scripts
- Provide sensible defaults for all parameters
- Allow configuration via files or arguments
- Document all configurable values"""

OUTPUT_FORMAT = """# OUTPUT FORMAT

Use this EXACT format for each file:

===FILE: relative/path/to/file.py===
<complete file contents>
===END_FILE===

RULES:
- Path must be relative (e.g., src/models/network.py, NOT /absolute/path)
- No markdown code blocks inside the file content
- Each file must be complete - no partial implementations
- Include all necessary imports at the top of each file"""

BATCH_INSTRUCTIONS = {
    "core": """# CURRENT TASK: Generate CORE Implementation Files

You are generating the foundational code. These files will be imported by everything else.

## What to Generate:
1. **Model/Algorithm Classes**: The main implementations from the paper
2. **Core Data Structures**: Custom types, containers, graph representations
3. **Mathematical Functions**: Loss functions, metrics, computations

## Naming Convention (IMPORTANT):
Establish clear, simple names that you will use consistently:
- Classes: PascalCase (e.g., `GraphNetwork`, `SATSolver`, `DataLoader`)
- Functions: snake_case (e.g., `compute_loss`, `forward_pass`)
- Constants: UPPER_SNAKE_CASE (e.g., `DEFAULT_LR`, `MAX_ITERATIONS`)

## Structure Each File:
```
# Imports (standard lib, third-party, local)
# Constants
# Helper functions
# Main classes
# Module-level convenience functions
```

## Self-Contained Design:
- Minimize dependencies between core files
- Each file should be importable independently
- Use dependency injection over hard-coded dependencies""",

    "support": """# CURRENT TASK: Generate SUPPORT Files

You are generating utilities that support the core implementation.

## What to Generate:
1. **Data Utilities**: Loading, parsing, preprocessing
2. **Configuration**: Config classes, YAML/JSON parsing
3. **Helpers**: Common operations, file I/O, logging setup

## CRITICAL - Import from Core Files:
You MUST import from the core files generated in the previous batch.
Check the "Previously Generated Files" list and use the EXACT class/function names.

Example - If core files defined:
- `src/models/network.py` with class `GraphNetwork`
- `src/solvers/dpll.py` with class `DPLLSolver`

Then import as:
```python
from src.models.network import GraphNetwork  # NOT GraphNeuralNetwork
from src.solvers.dpll import DPLLSolver      # NOT DPLL or Solver
```

## Make Utilities Robust:
- Handle missing files with clear error messages
- Validate data formats before processing
- Provide sensible defaults for all parameters""",

    "experiment": """# CURRENT TASK: Generate EXPERIMENT Files (Entry Points)

You are generating the runnable scripts. These MUST work out of the box.

## What to Generate:
1. **Training Script** (train.py or main.py): Full training pipeline
2. **Evaluation Script**: Test trained models
3. **Demo/Quick Test**: Simple script to verify installation works

## CRITICAL - Every Script Must Be Runnable:
```python
#!/usr/bin/env python3
\"\"\"Script description.\"\"\"

import argparse
# ... imports ...

def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/")
    # ... more args with DEFAULTS ...
    args = parser.parse_args()

    # Actual implementation here
    # NOT placeholders

if __name__ == "__main__":
    main()
```

## Handle Missing Dependencies Gracefully:
```python
try:
    from src.models.network import GraphNetwork
except ImportError as e:
    print(f"Error: Could not import required module: {e}")
    print("Make sure you've installed the package: pip install -e .")
    sys.exit(1)
```

## Provide Quick Verification:
Include a way to test the setup works:
```python
# In main(), add a --test flag:
if args.test:
    print("Running quick verification...")
    # Create small dummy data
    # Run one forward pass
    # Print "Setup verified successfully!"
    return
```""",

    "docs": """# CURRENT TASK: Generate DOCUMENTATION Files

## README.md Must Include:
1. **Quick Start** (copy-paste commands that work):
   ```bash
   pip install -r requirements.txt
   pip install -e .
   python train.py --test  # Verify setup
   ```

2. **Project Structure**: List main files and their purpose

3. **Usage Examples**: Actual working commands

4. **Requirements**: Python version, hardware needs

## requirements.txt:
- List ONLY packages actually imported in the code
- Pin major versions (e.g., torch>=2.0.0)
- Don't include packages that aren't used

## setup.py (if needed):
- Make the package installable with `pip install -e .`
- Include all subpackages"""
}

CONSISTENCY_REMINDER = """# FINAL REMINDER: CONSISTENCY IS EVERYTHING

The #1 reason generated code fails is inconsistent naming.

BEFORE YOU OUTPUT:
1. List all class names you're defining in this batch
2. List all class names you're importing from previous batches
3. Verify EVERY import and instantiation uses these EXACT names

If a previous file has `class DataProcessor`, you import `DataProcessor`.
NOT `DataPreprocessor`, NOT `Processor`, NOT `DataHandler`.

THE EXACT NAME. ALWAYS."""


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
        CRITICAL_REQUIREMENTS,
        batch_instruction,
        GENERATION_RULES,
        VERIFICATION_CHECKLIST,
        OUTPUT_FORMAT,
        CONSISTENCY_REMINDER,
    ])


# Default prompt for single-batch generation
CODE_GENERATION_PROMPT = get_code_generation_prompt("core")
