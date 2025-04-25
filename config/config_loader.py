import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.load_config()

    def load_config(self) -> None:
        """Load configuration from YAML file."""
        # Look for config.yaml in the main directory
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Error loading configuration: {str(e)}")

    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration dictionary."""
        return self._config

    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration."""
        return self._config.get('openai', {})

    def get_bigquery_config(self) -> Dict[str, Any]:
        """Get BigQuery configuration."""
        return self._config.get('bigquery', {})

    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration."""
        return self._config.get('app', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})

    def get_credentials_config(self) -> Dict[str, Any]:
        """Get credentials configuration."""
        return self._config.get('credentials', {}) 