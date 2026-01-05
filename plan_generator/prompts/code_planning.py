"""Code planning prompt sections and assembled prompt."""

INTRO = (
    "You are creating a DETAILED, COMPLETE reproduction plan by integrating "
    "comprehensive analysis results."
)

INPUT_SECTION = """# INPUT
You receive two exhaustive analyses:
1. **Comprehensive Paper Analysis**: Complete paper structure, components, and requirements
2. **Complete Algorithm Extraction**: All algorithms, formulas, pseudocode, and technical details"""

OBJECTIVE = """# OBJECTIVE
Create an implementation plan so detailed that a developer can reproduce the ENTIRE paper without reading it."""

OUTPUT_REQUIREMENT = """# CRITICAL: COMPLETE OUTPUT REQUIREMENT
MANDATORY: You MUST generate ALL 5 sections completely. DO NOT stop early or truncate any section.

## Output Completeness Strategy:
**Your #1 Priority**: Ensure ALL 5 sections are present and complete before finishing your response.

## Content Balance Guidelines (STRICTLY FOLLOW):
- **Section 1 (File Structure)**: ~800-1000 chars - Brief file listing with priority order
- **Section 2 (Implementation Components)**: ~3000-4000 chars - CORE section with all algorithms/components
- **Section 3 (Validation)**: ~2000-2500 chars - Experiments and expected results
- **Section 4 (Environment)**: ~800-1000 chars - Dependencies and requirements
- **Section 5 (Implementation Strategy)**: ~1500-2000 chars - Step-by-step approach

**Total Target**: 8000-10000 characters for complete plan

**Self-Check Before Finishing**:
- Did you include file_structure section?
- Did you include implementation_components section?
- Did you include validation_approach section?
- Did you include environment_setup section?
- Did you include implementation_strategy section?
- If ANY answer is NO, continue writing until ALL sections are complete!"""

FILE_PRIORITY_GUIDELINES = """## File Priority Guidelines:
**Implementation Priority Order**:
1. **FIRST**: Core algorithm/model files (highest priority)
2. **SECOND**: Supporting modules and utilities
3. **THIRD**: Experiment and evaluation scripts
4. **FOURTH**: Configuration and data handling
5. **LAST**: Documentation files (README.md, requirements.txt) - These should be created AFTER core implementation

Note: README and requirements.txt are maintenance files that depend on the final implementation, so plan them last but INCLUDE them in the file structure."""

SYNTHESIS_PROCESS = """# DETAILED SYNTHESIS PROCESS

## 1. MERGE ALL INFORMATION
Combine EVERYTHING from both analyses:
- Every algorithm with its pseudocode
- Every component with its architecture
- Every hyperparameter with its value
- Every experiment with expected results

## 2. MAP CONTENT TO IMPLEMENTATION

For each component you identify, specify how it will be implemented:

```
# DESIGN YOUR MAPPING: Connect paper content to code organization
[For each algorithm/component/method in the paper]:
  - What it does and where it's described in the paper
  - How you'll organize the code (files, classes, functions - your choice)
  - What specific formulas, algorithms, or procedures need implementation
  - Dependencies and relationships with other components
  - Implementation approach that makes sense for this specific paper
```

## 3. EXTRACT ALL TECHNICAL DETAILS

Identify every technical detail that needs implementation:

```
# COMPREHENSIVE TECHNICAL EXTRACTION:
[Gather all implementation-relevant details from the paper]:
  - All algorithms with complete pseudocode and mathematical formulations
  - All parameters, hyperparameters, and configuration values
  - All architectural details (if applicable to your paper type)
  - All experimental procedures and evaluation methods
  - Any implementation hints, tricks, or special considerations mentioned
```
"""

OUTPUT_FORMAT = """# COMPREHENSIVE OUTPUT FORMAT

```yaml
complete_reproduction_plan:
  paper_info:
    title: "[Full paper title]"
    core_contribution: "[Main innovation being reproduced]"

  # SECTION 1: File Structure Design

  # DESIGN YOUR OWN STRUCTURE: Create a file organization that best serves this specific paper
  # - Analyze what the paper contains (algorithms, models, experiments, systems, etc.)
  # - Organize files and directories in the most logical way for implementation
  # - Create meaningful names and groupings based on paper content
  # - Keep it clean, intuitive, and focused on what actually needs to be implemented
  # - INCLUDE documentation files (README.md, requirements.txt) but mark them for LAST implementation

  file_structure: |
    [Design and specify your own project structure here - KEEP THIS BRIEF]
    [Include ALL necessary files including README.md and requirements.txt]
    [Organize based on what this paper actually contains and needs]
    [Create directories and files that make sense for this specific implementation]
    [IMPORTANT: Include executable files (e.g., main.py, run.py, train.py, demo.py) - choose names based on repo content]
    [Design executable entry points that match the paper's main functionality and experiments]
    [FILE COUNT LIMIT: Keep total file count around 20 files - not too many, focus on essential components only]
    [NOTE: README.md and requirements.txt should be implemented LAST after all code files]

  # SECTION 2: Implementation Components

  # IDENTIFY AND SPECIFY: What needs to be implemented based on this paper
  # - List all algorithms, models, systems, or components mentioned
  # - Map each to implementation details and file locations
  # - Include formulas, pseudocode, and technical specifications
  # - Organize in whatever way makes sense for this paper

  implementation_components: |
    [List and specify all components that need implementation]
    [For each component: purpose, location, algorithms, formulas, technical details]
    [Organize and structure this based on the paper's actual content]

  # SECTION 3: Validation & Evaluation

  # DESIGN VALIDATION: How to verify the implementation works correctly
  # - Define what experiments, tests, or proofs are needed
  # - Specify expected results from the paper (figures, tables, theorems)
  # - Design validation approach appropriate for this paper's domain
  # - Include setup requirements and success criteria

  validation_approach: |
    [Design validation strategy appropriate for this paper]
    [Specify experiments, tests, or mathematical verification needed]
    [Define expected results and success criteria]
    [Include any special setup or evaluation requirements]

  # SECTION 4: Environment & Dependencies

  # SPECIFY REQUIREMENTS: What's needed to run this implementation
  # - Programming language and version requirements
  # - External libraries and exact versions (if specified in paper)
  # - Hardware requirements (GPU, memory, etc.)
  # - Any special setup or installation steps

  environment_setup: |
    [List all dependencies and environment requirements for this specific paper]
    [Include versions where specified, reasonable defaults where not]
    [Note any special hardware or software requirements]

  # SECTION 5: Implementation Strategy

  # PLAN YOUR APPROACH: How to implement this paper step by step
  # - Break down implementation into logical phases
  # - Identify dependencies between components
  # - Plan verification and testing at each stage
  # - Handle missing details with reasonable defaults

  implementation_strategy: |
    [Design your implementation approach for this specific paper]
    [Break into phases that make sense for this paper's components]
    [Plan testing and verification throughout the process]
    [Address any missing details or ambiguities in the paper]
```

BE EXHAUSTIVE. Every algorithm, every formula, every parameter, every file should be specified in complete detail."""

CODE_PLANNING_PROMPT = "\n\n".join(
    [
        INTRO,
        INPUT_SECTION,
        OBJECTIVE,
        OUTPUT_REQUIREMENT,
        FILE_PRIORITY_GUIDELINES,
        SYNTHESIS_PROCESS,
        OUTPUT_FORMAT,
    ]
)
