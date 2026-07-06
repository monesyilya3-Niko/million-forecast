"""Value betting filter for football predictions.

Implements EV calculation, Kelly Criterion, and risk management
to identify profitable betting opportunities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValueBet:
    """A value betting opportunity."""

    match_id: str
    home_team: str
    away_team: str
    outcome: str  # "home_win", "draw", "away_win"

    # Probabilities
    model_probability: float
    market_probability: float

    # Odds
    odds: float

    # Value metrics
    ev: float  # Expected Value
    edge: float  # Model - Market probability
    kelly_fraction: float  # Kelly Criterion fraction

    # Risk assessment
    confidence: str  # "high", "medium", "low"
    risk_score: float  # 0-1, higher = more risky


class ValueBettingFilter:
    """Filter for identifying value betting opportunities.

    Uses Expected Value (EV) and Kelly Criterion to identify
    bets with positive expected value.
    """

    def __init__(
        self,
        min_ev: float = 0.05,  # Minimum 5% EV
        min_edge: float = 0.03,  # Minimum 3% edge
        max_kelly: float = 0.10,  # Maximum 10% Kelly fraction
        min_odds: float = 1.20,  # Minimum odds
        max_odds: float = 10.0,  # Maximum odds
        kelly_fraction: float = 0.25,  # Fraction of Kelly to use (quarter Kelly)
    ) -> None:
        self.min_ev = min_ev
        self.min_edge = min_edge
        self.max_kelly = max_kelly
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.kelly_fraction = kelly_fraction

    def filter(
        self,
        matches: list[dict],
    ) -> list[ValueBet]:
        """Filter matches for value betting opportunities.

        Args:
            matches: List of match dictionaries with predictions and odds

        Returns:
            List of ValueBet opportunities
        """
        value_bets = []

        for match in matches:
            bets = self._analyze_match(match)
            value_bets.extend(bets)

        # Sort by EV (descending)
        value_bets.sort(key=lambda x: x.ev, reverse=True)

        logger.info(f"Found {len(value_bets)} value bets from {len(matches)} matches")
        return value_bets

    def _analyze_match(self, match: dict) -> list[ValueBet]:
        """Analyze a single match for value bets."""
        bets = []

        match_id = match.get("match_id", "unknown")
        home_team = match.get("home_team", "Unknown")
        away_team = match.get("away_team", "Unknown")

        # Check each outcome
        for outcome in ["home_win", "draw", "away_win"]:
            model_prob = match.get(f"model_{outcome}", 0)
            odds = match.get(f"odds_{outcome}", 0)

            if odds <= 1.0 or model_prob <= 0:
                continue

            # Calculate market probability (implied by odds)
            market_prob = 1.0 / odds

            # Calculate EV
            ev = (model_prob * odds) - 1

            # Calculate edge
            edge = model_prob - market_prob

            # Calculate Kelly fraction
            kelly = self._kelly_fraction(model_prob, odds)

            # Assess confidence and risk
            confidence, risk_score = self._assess_confidence(model_prob, market_prob, ev, edge, kelly)

            # Apply filters
            if not self._passes_filters(ev, edge, kelly, odds, confidence):
                continue

            value_bet = ValueBet(
                match_id=match_id,
                home_team=home_team,
                away_team=away_team,
                outcome=outcome,
                model_probability=model_prob,
                market_probability=market_prob,
                odds=odds,
                ev=ev,
                edge=edge,
                kelly_fraction=kelly,
                confidence=confidence,
                risk_score=risk_score,
            )

            bets.append(value_bet)

        return bets

    def _kelly_fraction(self, probability: float, odds: float) -> float:
        """Calculate Kelly Criterion fraction.

        Kelly % = (bp - q) / b
        where b = odds - 1, p = probability, q = 1 - p
        """
        b = odds - 1
        p = probability
        q = 1 - p

        if b <= 0:
            return 0.0

        kelly = (b * p - q) / b

        # Apply fractional Kelly (more conservative)
        kelly = kelly * self.kelly_fraction

        # Cap at maximum
        kelly = min(kelly, self.max_kelly)

        return max(0.0, kelly)

    def _assess_confidence(
        self,
        model_prob: float,
        market_prob: float,
        ev: float,
        edge: float,
        kelly: float,
    ) -> tuple[str, float]:
        """Assess confidence level and risk score."""
        # Risk score (0-1, higher = more risky)
        risk_factors = []

        # Factor 1: Model uncertainty (higher probability = more certain)
        risk_factors.append(1.0 - model_prob)

        # Factor 2: Market disagreement (larger edge = more uncertain)
        risk_factors.append(min(abs(edge) * 10, 1.0))

        # Factor 3: Odds level (higher odds = more variance)
        risk_factors.append(min((1.0 / market_prob) / 10, 1.0))

        risk_score = float(np.mean(risk_factors))

        # Confidence level
        if ev >= 0.15 and edge >= 0.08 and model_prob >= 0.40:
            confidence = "high"
        elif ev >= 0.08 and edge >= 0.05:
            confidence = "medium"
        else:
            confidence = "low"

        return confidence, risk_score

    def _passes_filters(
        self,
        ev: float,
        edge: float,
        kelly: float,
        odds: float,
        confidence: str,
    ) -> bool:
        """Check if bet passes all filters."""
        # EV filter
        if ev < self.min_ev:
            return False

        # Edge filter
        if edge < self.min_edge:
            return False

        # Odds filter
        if odds < self.min_odds or odds > self.max_odds:
            return False

        # Kelly filter
        return not kelly <= 0


def format_value_bet(bet: ValueBet) -> str:
    """Format a value bet for display."""
    outcome_labels = {
        "home_win": "主胜",
        "draw": "平局",
        "away_win": "客胜",
    }

    return (
        f"{bet.home_team} vs {bet.away_team}\n"
        f"  推荐: {outcome_labels[bet.outcome]} @ {bet.odds:.2f}\n"
        f"  模型概率: {bet.model_probability:.1%} | 市场概率: {bet.market_probability:.1%}\n"
        f"  EV: {bet.ev:+.1%} | 边际: {bet.edge:+.1%}\n"
        f"  Kelly: {bet.kelly_fraction:.1%} | 信心: {bet.confidence}\n"
    )
