"""Загрузка и валидация конфигов объектов."""

from .loader import ConfigError, load_config, to_dcf_inputs
from .schema import ObjectConfig

__all__ = ["ConfigError", "load_config", "to_dcf_inputs", "ObjectConfig"]
