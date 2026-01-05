"""
LLM utility functions for the plan_generator module.

Provides functions for configuration and LLM settings.
"""

import yaml
from typing import Tuple, Dict, List
from pathlib import Path

from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM


def _get_module_dir() -> Path:
    """Get the directory where this module is located."""
    return Path(__file__).parent


def _load_config() -> dict:
    """Load the config file once and return it."""
    config_path = _get_module_dir() / "mcp_agent.config.yaml"
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error reading config: {e}")
    return {}


def get_llm_class():
    """
    Get the LLM class to use for workflows.

    Returns:
        The OpenAIAugmentedLLM class for use as llm_factory.
    """
    return OpenAIAugmentedLLM


def get_token_limits() -> Tuple[int, int]:
    """
    Get token limits from mcp_agent.config.yaml.

    Returns:
        tuple: (base_max_tokens, retry_max_tokens)
    """
    config = _load_config()
    openai_config = config.get("openai", {})

    base_tokens = openai_config.get("base_max_tokens", 40000)
    retry_tokens = openai_config.get("retry_max_tokens", 32768)

    return base_tokens, retry_tokens


def get_llm_params() -> Tuple[float, int]:
    """
    Get LLM request parameters from config.

    Returns:
        tuple: (temperature, max_iterations)
    """
    config = _load_config()
    llm_params = config.get("llm_params", {})

    temperature = llm_params.get("temperature", 0.3)
    max_iterations = llm_params.get("max_iterations", 10)

    return temperature, max_iterations


def get_tracing_config() -> Tuple[bool, str, str]:
    """
    Get tracing configuration from config.

    Returns:
        tuple: (enabled, project_name, endpoint)
    """
    config = _load_config()
    tracing = config.get("tracing", {})

    enabled = tracing.get("enabled", True)
    project_name = tracing.get("project_name", "plan-generator")
    endpoint = tracing.get("endpoint", "http://localhost:6006/v1/traces")

    return enabled, project_name, endpoint


def get_agent_servers() -> Dict[str, List[str]]:
    """
    Get agent server configurations from config.

    Returns:
        dict: mapping of agent name to list of server names
    """
    config = _load_config()
    agent_servers = config.get("agent_servers", {})

    return {
        "concept_analysis": agent_servers.get("concept_analysis", []),
        "algorithm_analysis": agent_servers.get("algorithm_analysis", []),
        "code_planner": agent_servers.get("code_planner", []),
    }
