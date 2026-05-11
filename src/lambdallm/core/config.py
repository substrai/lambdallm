"""Configuration loader for LambdaLLM.

Loads lambdallm.yaml and provides typed access to all settings.
Implements Convention Over Configuration - everything has sensible defaults.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from lambdallm.core.exceptions import ConfigurationError

logger = logging.getLogger("lambdallm")

DEFAULT_CONFIG = {
    "project": {
        "name": "my-lambdallm-app",
        "version": "0.1.0",
        "runtime": "python3.12",
    },
    "defaults": {
        "model": "bedrock/claude-3-haiku",
        "region": "us-east-1",
        "timeout": 30,
        "memory": 256,
        "state": {
            "provider": "dynamodb",
            "table_prefix": "lambdallm-",
            "ttl": 86400,
        },
    },
    "models": {
        "fast": {
            "provider": "bedrock",
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "max_tokens": 1000,
            "temperature": 0.3,
        },
        "smart": {
            "provider": "bedrock",
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "max_tokens": 4000,
            "temperature": 0.7,
        },
        "powerful": {
            "provider": "bedrock",
            "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "max_tokens": 8000,
            "temperature": 0.7,
        },
    },
    "cost": {
        "budget": {"daily": 50.0, "monthly": 1000.0},
        "on_budget_exceeded": "downgrade",
        "routing": {"strategy": "cost-optimized", "quality_threshold": 0.85},
    },
    "observability": {
        "tracing": "xray",
        "metrics": "cloudwatch",
        "log_level": "INFO",
        "log_prompts": False,
    },
    "environments": {},
}


@dataclass
class ProjectConfig:
    name: str = "my-lambdallm-app"
    version: str = "0.1.0"
    runtime: str = "python3.12"


@dataclass
class StateConfig:
    provider: str = "dynamodb"
    table_prefix: str = "lambdallm-"
    ttl: int = 86400


@dataclass
class DefaultsConfig:
    model: str = "bedrock/claude-3-haiku"
    region: str = "us-east-1"
    timeout: int = 30
    memory: int = 256
    state: StateConfig = field(default_factory=StateConfig)


@dataclass
class BudgetConfig:
    daily: float = 50.0
    monthly: float = 1000.0


@dataclass
class RoutingConfig:
    strategy: str = "cost-optimized"
    quality_threshold: float = 0.85


@dataclass
class CostConfig:
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    on_budget_exceeded: str = "downgrade"
    routing: RoutingConfig = field(default_factory=RoutingConfig)


@dataclass
class ObservabilityConfig:
    tracing: str = "xray"
    metrics: str = "cloudwatch"
    log_level: str = "INFO"
    log_prompts: bool = False


@dataclass
class LambdaLLMConfig:
    """Root configuration object for LambdaLLM."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    models: dict = field(default_factory=lambda: DEFAULT_CONFIG["models"])
    cost: CostConfig = field(default_factory=CostConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    environments: dict = field(default_factory=dict)
    _raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: Optional[str] = None, env: Optional[str] = None) -> "LambdaLLMConfig":
        """Load configuration from lambdallm.yaml.

        Searches in order:
        1. Explicit path argument
        2. LAMBDALLM_CONFIG env var
        3. ./lambdallm.yaml (current directory)
        4. Falls back to defaults (Convention Over Configuration)
        """
        config_path = cls._find_config(path)

        if config_path and config_path.exists():
            raw = cls._load_yaml(config_path)
            logger.info(f"Loaded config from: {config_path}")
        else:
            raw = DEFAULT_CONFIG.copy()
            logger.info("Using default configuration (no lambdallm.yaml found)")

        # Apply environment overrides
        if env and "environments" in raw and env in raw["environments"]:
            raw = cls._merge_env(raw, raw["environments"][env])

        return cls._from_dict(raw)

    @classmethod
    def _find_config(cls, explicit_path: Optional[str] = None) -> Optional[Path]:
        """Find the config file."""
        if explicit_path:
            return Path(explicit_path)

        env_path = os.environ.get("LAMBDALLM_CONFIG")
        if env_path:
            return Path(env_path)

        local_path = Path("lambdallm.yaml")
        if local_path.exists():
            return local_path

        return None

    @classmethod
    def _load_yaml(cls, path: Path) -> dict:
        """Load and parse YAML file."""
        try:
            import yaml
        except ImportError:
            raise ConfigurationError(
                "PyYAML required to load config: pip install pyyaml"
            )

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ConfigurationError(f"Invalid config file: {path}")

        return data

    @classmethod
    def _merge_env(cls, base: dict, env_overrides: dict) -> dict:
        """Deep merge environment overrides into base config."""
        import copy

        result = copy.deepcopy(base)
        for key, value in env_overrides.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = cls._merge_env(result[key], value)
            else:
                result[key] = value
        return result

    @classmethod
    def _from_dict(cls, raw: dict) -> "LambdaLLMConfig":
        """Create config object from raw dict."""
        project_data = raw.get("project", {})
        defaults_data = raw.get("defaults", {})
        cost_data = raw.get("cost", {})
        obs_data = raw.get("observability", {})

        state_data = defaults_data.get("state", {})

        return cls(
            project=ProjectConfig(
                name=project_data.get("name", "my-lambdallm-app"),
                version=project_data.get("version", "0.1.0"),
                runtime=project_data.get("runtime", "python3.12"),
            ),
            defaults=DefaultsConfig(
                model=defaults_data.get("model", "bedrock/claude-3-haiku"),
                region=defaults_data.get("region", "us-east-1"),
                timeout=defaults_data.get("timeout", 30),
                memory=defaults_data.get("memory", 256),
                state=StateConfig(
                    provider=state_data.get("provider", "dynamodb"),
                    table_prefix=state_data.get("table_prefix", "lambdallm-"),
                    ttl=state_data.get("ttl", 86400),
                ),
            ),
            models=raw.get("models", DEFAULT_CONFIG["models"]),
            cost=CostConfig(
                budget=BudgetConfig(
                    daily=cost_data.get("budget", {}).get("daily", 50.0),
                    monthly=cost_data.get("budget", {}).get("monthly", 1000.0),
                ),
                on_budget_exceeded=cost_data.get("on_budget_exceeded", "downgrade"),
                routing=RoutingConfig(
                    strategy=cost_data.get("routing", {}).get("strategy", "cost-optimized"),
                    quality_threshold=cost_data.get("routing", {}).get("quality_threshold", 0.85),
                ),
            ),
            observability=ObservabilityConfig(
                tracing=obs_data.get("tracing", "xray"),
                metrics=obs_data.get("metrics", "cloudwatch"),
                log_level=obs_data.get("log_level", "INFO"),
                log_prompts=obs_data.get("log_prompts", False),
            ),
            environments=raw.get("environments", {}),
            _raw=raw,
        )
