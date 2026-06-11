"""Загрузка и валидация конфигов объектов."""

from .loader import ConfigError, load_config, to_dcf_inputs, validate_config
from .schema import ObjectConfig

__all__ = ["ConfigError", "load_config", "validate_config", "to_dcf_inputs", "ObjectConfig"]
