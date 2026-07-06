"""Trainable model interfaces and implementations."""

from .base import FootballProbabilityModel
from .calibration import ProbabilityCalibrator
from .dixon_coles import DixonColesModel
from .ensemble import EnsembleModel
from .neural_net import NeuralNetModel
from .poisson import PoissonModel
from .team_intelligence import MatchIntelligence, TeamSnapshot, build_match_intelligence
from .xgboost_model import XGBoostModel

__all__ = [
    "DixonColesModel",
    "EnsembleModel",
    "FootballProbabilityModel",
    "MatchIntelligence",
    "NeuralNetModel",
    "PoissonModel",
    "ProbabilityCalibrator",
    "TeamSnapshot",
    "XGBoostModel",
    "build_match_intelligence",
]
