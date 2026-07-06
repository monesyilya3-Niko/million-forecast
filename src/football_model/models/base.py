from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)


class FootballProbabilityModel(ABC):
    """Stable interface shared by statistical and machine-learning models."""

    @abstractmethod
    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> FootballProbabilityModel:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def save(self, artifact_path: str) -> None:
        raise NotImplementedError
