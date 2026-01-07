"""
Plan Generator - Standalone module for research paper code analysis.

This module analyzes research papers and generates comprehensive YAML
implementation plans using a multi-agent architecture.

Usage:
    # Option 1: Run as module
    python -m plan_generator.workflow --paper-dir ./papers/my_paper

    # Option 2: Use programmatically
    from plan_generator import app
    import asyncio

    async def main():
        async with app.run() as running_app:
            result = await running_app.executor.execute_workflow(
                "analyze_paper",
                "./papers/my_paper"
            )
            print(result)

    asyncio.run(main())

    # Option 3: Use the legacy run_code_analyzer function (asyncio only)
    from plan_generator import run_code_analyzer
    import asyncio
    import logging

    async def main():
        result = await run_code_analyzer("./papers/my_paper", logging.getLogger())
        print(result)

    asyncio.run(main())

The execution engine (asyncio or temporal) is determined by the
execution_engine setting in mcp_agent.config.yaml. Simply change
the setting to switch between engines - no code changes needed.

IMPORTANT: When using temporal engine, you must run a worker first:
    python -m plan_generator.run_worker

The worker processes workflow tasks. For asyncio, no worker is needed.

The module uses three specialized agents:
- ConceptAnalysisAgent: Analyzes paper structure and architecture
- AlgorithmAnalysisAgent: Extracts algorithms, formulas, and technical details
- CodePlannerAgent: Generates comprehensive implementation plan

Configuration:
    - mcp_agent.config.yaml: execution_engine (asyncio/temporal), LLM settings
    - mcp_agent.secrets.yaml: API keys
"""

# Main workflow interface (works with both asyncio and temporal)
from .workflow import app, run_workflow

# Code generation workflow
from .code_generator import run_code_generation, code_gen_app

# Legacy asyncio-only interface
#from .analyzer import run_code_analyzer

__all__ = [
    "app",
    "run_workflow",
    "run_code_generation",
    "code_gen_app",
#    "run_code_analyzer",
]
__version__ = "1.0.0"
