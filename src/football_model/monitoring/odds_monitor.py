"""Real-time odds monitoring and anomaly detection.

Tracks odds movements, detects sharp money, and provides
alerts for significant line movements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OddsMovement:
    """Tracks odds movement for a single market."""

    match_id: str
    market: str  # "HAD" or "HHAD"
    selection: str  # "H", "D", "A"

    # Current state
    current_odds: float
    opening_odds: float

    # Movement
    odds_change: float
    odds_change_pct: float
    implied_prob_change: float

    # Timing
    first_seen: datetime
    last_seen: datetime
    snapshot_count: int

    # Volatility
    odds_std: float
    odds_volatility: float  # Coefficient of variation

    # Direction
    direction: str  # "steam_move", "drift", "stable"
    steam_detected: bool


@dataclass(frozen=True)
class OddsAlert:
    """Alert for significant odds movement."""

    match_id: str
    alert_type: str  # "steam_move", "reverse_line_movement", "odds_spike"
    severity: str  # "high", "medium", "low"
    message: str
    details: dict


class OddsMonitor:
    """Monitor odds movements and detect anomalies.

    Tracks odds snapshots and identifies significant movements
    that may indicate sharp money or market inefficiencies.
    """

    def __init__(
        self,
        steam_threshold: float = 0.05,  # 5% probability shift
        volatility_threshold: float = 0.03,  # 3% coefficient of variation
        min_snapshots: int = 3,  # Minimum snapshots for analysis
        lookback_hours: int = 24,  # Lookback period
    ) -> None:
        self.steam_threshold = steam_threshold
        self.volatility_threshold = volatility_threshold
        self.min_snapshots = min_snapshots
        self.lookback_hours = lookback_hours

    def analyze_movements(
        self,
        odds_history: pd.DataFrame,
    ) -> tuple[list[OddsMovement], list[OddsAlert]]:
        """Analyze odds movements from history.

        Args:
            odds_history: DataFrame with columns:
                - match_id, market, selection, odds, captured_at

        Returns:
            Tuple of (movements, alerts)
        """
        if odds_history.empty:
            return [], []

        movements = []
        alerts = []

        # Group by match/market/selection
        grouped = odds_history.groupby(["match_id", "market", "selection"])

        for (match_id, market, selection), group in grouped:
            if len(group) < self.min_snapshots:
                continue

            # Sort by time
            group = group.sort_values("captured_at")

            # Analyze this market
            movement = self._analyze_single_market(match_id, market, selection, group)
            movements.append(movement)

            # Check for alerts
            market_alerts = self._check_alerts(movement)
            alerts.extend(market_alerts)

        logger.info(f"Analyzed {len(movements)} markets, found {len(alerts)} alerts")
        return movements, alerts

    def _analyze_single_market(
        self,
        match_id: str,
        market: str,
        selection: str,
        history: pd.DataFrame,
    ) -> OddsMovement:
        """Analyze odds movement for a single market."""
        odds_series = history["odds"].astype(float)
        times = pd.to_datetime(history["captured_at"])

        # Basic stats
        opening_odds = float(odds_series.iloc[0])
        current_odds = float(odds_series.iloc[-1])
        odds_change = current_odds - opening_odds
        odds_change_pct = odds_change / opening_odds

        # Implied probability change
        opening_prob = 1.0 / opening_odds
        current_prob = 1.0 / current_odds
        implied_prob_change = current_prob - opening_prob

        # Volatility
        odds_std = float(odds_series.std())
        odds_volatility = odds_std / odds_series.mean() if odds_series.mean() > 0 else 0

        # Detect steam moves (rapid, significant movement)
        steam_detected = self._detect_steam_move(odds_series, times)

        # Determine direction
        if steam_detected:
            direction = "steam_move"
        elif abs(implied_prob_change) > self.steam_threshold:
            direction = "drift"
        else:
            direction = "stable"

        return OddsMovement(
            match_id=match_id,
            market=market,
            selection=selection,
            current_odds=current_odds,
            opening_odds=opening_odds,
            odds_change=odds_change,
            odds_change_pct=odds_change_pct,
            implied_prob_change=implied_prob_change,
            first_seen=times.iloc[0],
            last_seen=times.iloc[-1],
            snapshot_count=len(history),
            odds_std=odds_std,
            odds_volatility=odds_volatility,
            direction=direction,
            steam_detected=steam_detected,
        )

    def _detect_steam_move(
        self,
        odds: pd.Series,
        times: pd.Series,
    ) -> bool:
        """Detect steam moves (rapid, significant movement).

        Steam moves typically:
        1. Happen quickly (< 2 hours)
        2. Move significantly (> 3% probability shift)
        3. Are one-directional
        """
        if len(odds) < 3:
            return False

        # Calculate time span
        time_span = (times.iloc[-1] - times.iloc[0]).total_seconds() / 3600

        if time_span > 2:  # More than 2 hours
            return False

        # Calculate probability shift
        opening_prob = 1.0 / float(odds.iloc[0])
        current_prob = 1.0 / float(odds.iloc[-1])
        prob_shift = abs(current_prob - opening_prob)

        if prob_shift < self.steam_threshold:
            return False

        # Check if movement is one-directional
        diffs = odds.diff().dropna()
        if len(diffs) < 2:
            return False

        # All moves should be in same direction
        all_positive = (diffs > 0).all()
        all_negative = (diffs < 0).all()

        return all_positive or all_negative

    def _check_alerts(self, movement: OddsMovement) -> list[OddsAlert]:
        """Check for alert conditions."""
        alerts = []

        # Steam move alert
        if movement.steam_detected:
            alerts.append(
                OddsAlert(
                    match_id=movement.match_id,
                    alert_type="steam_move",
                    severity="high",
                    message=f"Steam move detected: {movement.selection} odds moved {movement.odds_change_pct:+.1%}",
                    details={
                        "opening_odds": movement.opening_odds,
                        "current_odds": movement.current_odds,
                        "probability_shift": movement.implied_prob_change,
                    },
                )
            )

        # Large probability shift
        if abs(movement.implied_prob_change) > self.steam_threshold * 2:
            alerts.append(
                OddsAlert(
                    match_id=movement.match_id,
                    alert_type="large_shift",
                    severity="medium",
                    message=f"Large odds shift: {movement.selection} probability changed {movement.implied_prob_change:+.1%}",
                    details={
                        "probability_change": movement.implied_prob_change,
                        "odds_change": movement.odds_change,
                    },
                )
            )

        # High volatility
        if movement.odds_volatility > self.volatility_threshold:
            alerts.append(
                OddsAlert(
                    match_id=movement.match_id,
                    alert_type="high_volatility",
                    severity="low",
                    message=f"High volatility: {movement.selection} CV={movement.odds_volatility:.2%}",
                    details={
                        "volatility": movement.odds_volatility,
                        "std": movement.odds_std,
                    },
                )
            )

        return alerts


def generate_movement_report(
    movements: list[OddsMovement],
    alerts: list[OddsAlert],
) -> str:
    """Generate a human-readable movement report."""
    report = "=== 赔率变动监控报告 ===\n\n"

    # Summary
    report += f"监控市场数: {len(movements)}\n"
    report += f"警报数: {len(alerts)}\n\n"

    # Alerts
    if alerts:
        report += "--- 警报 ---\n"
        for alert in alerts:
            report += f"[{alert.severity.upper()}] {alert.message}\n"
        report += "\n"

    # Steam moves
    steam_moves = [m for m in movements if m.steam_detected]
    if steam_moves:
        report += "--- Steam Moves ---\n"
        for move in steam_moves:
            report += (
                f"{move.match_id}: {move.selection} "
                f"{move.opening_odds:.2f} -> {move.current_odds:.2f} "
                f"({move.implied_prob_change:+.1%})\n"
            )
        report += "\n"

    # Large movements
    large_moves = [m for m in movements if abs(m.implied_prob_change) > 0.03 and not m.steam_detected]
    if large_moves:
        report += "--- 大幅变动 ---\n"
        for move in large_moves:
            report += (
                f"{move.match_id}: {move.selection} "
                f"{move.opening_odds:.2f} -> {move.current_odds:.2f} "
                f"({move.implied_prob_change:+.1%})\n"
            )

    return report
