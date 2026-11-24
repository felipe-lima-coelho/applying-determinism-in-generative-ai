"""Portas internas substituíveis da extração."""

from .dataset_repository import DatasetRepository
from .inference import InferenceError, InferenceProvider

__all__ = ["DatasetRepository", "InferenceError", "InferenceProvider"]
