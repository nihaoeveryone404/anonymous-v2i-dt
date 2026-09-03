"""Model definitions shared by training and inference."""

from .cpfl_models import PFL_KPIPredictor, PFL_REMNet, SimpleExtractor

__all__ = ["SimpleExtractor", "PFL_REMNet", "PFL_KPIPredictor"]
