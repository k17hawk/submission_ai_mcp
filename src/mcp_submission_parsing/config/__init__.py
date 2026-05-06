from pathlib import Path
import yaml
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """Load field definitions from YAML config file"""
    config_path = Path(__file__).parent / 'field_definitions.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_patterns() -> Dict[str, Any]:
    """Get regex patterns from config"""
    config = load_config()
    return config.get('patterns', {})


def get_normalizer_config() -> Dict[str, Any]:
    """Get normalizer configuration (mappings + valid values)"""
    config = load_config()
    return {
        'mappings': config.get('mappings', {}),
        'valid_values': config.get('valid_values', {}),
        'auto_makes': config.get('auto_makes', []),
    }