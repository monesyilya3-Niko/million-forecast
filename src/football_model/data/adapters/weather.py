"""Weather data adapter using Open-Meteo free API.

Fetches weather forecasts for match venues to include as features.
Open-Meteo is free, no API key required, and provides hourly forecasts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)

# Major venue coordinates for common leagues
VENUE_COORDS: dict[str, tuple[float, float]] = {
    # Swedish Allsvenskan venues
    "Stockholm": (59.33, 18.07),
    "Gothenburg": (57.71, 11.97),
    "Malmö": (55.60, 13.00),
    "Solna": (59.36, 18.00),
    "Borås": (57.72, 12.94),
    "Norrköping": (58.59, 16.18),
    "Helsingborg": (56.05, 12.70),
    "Kalmar": (56.66, 16.36),
    "Mjällby": (56.05, 14.73),
    "Värnamo": (57.19, 14.05),
    "Uppsala": (59.86, 17.64),
    "Gävle": (60.67, 17.15),
    "Sundsvall": (62.39, 17.31),
    "Örebro": (59.27, 15.21),
    # World Cup 2026 venues (USA)
    "East Rutherford": (40.81, -74.07),
    "Mexico City": (19.43, -99.13),
    "Los Angeles": (34.05, -118.24),
    "Dallas": (32.78, -96.80),
    "Miami": (25.76, -80.19),
    "Houston": (29.76, -95.37),
    "Philadelphia": (39.95, -75.17),
    "Kansas City": (39.10, -94.58),
    "Atlanta": (33.75, -84.39),
    "Seattle": (47.61, -122.33),
    "San Francisco": (37.77, -122.42),
    "Boston": (42.36, -71.06),
    "Vancouver": (49.28, -123.12),
    "Toronto": (43.65, -79.38),
    "Guadalajara": (20.67, -103.35),
    "Monterrey": (25.67, -100.32),
    # England
    "London": (51.51, -0.13),
    "Manchester": (53.48, -2.24),
    "Liverpool": (53.41, -2.98),
    "Birmingham": (52.49, -1.89),
    "Leeds": (53.80, -1.55),
    "Newcastle": (54.98, -1.62),
    "Leicester": (52.63, -1.13),
    "Brighton": (50.83, -0.14),
    "Southampton": (50.90, -1.40),
    "Nottingham": (52.95, -1.15),
}


@dataclass(frozen=True)
class WeatherData:
    """Weather data for a match."""

    temperature_c: float
    humidity_pct: float
    wind_speed_kmh: float
    precipitation_mm: float
    cloud_cover_pct: float
    weather_code: int
    description: str


# WMO weather codes
WEATHER_CODES: dict[int, str] = {
    0: "晴",
    1: "大部晴",
    2: "多云",
    3: "阴天",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中毛毛雨",
    55: "大毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "阵雨",
    81: "中阵雨",
    82: "大阵雨",
    95: "雷暴",
    96: "雷暴+小冰雹",
    99: "雷暴+大冰雹",
}


class WeatherAdapter:
    """Fetch weather forecasts from Open-Meteo."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def fetch_forecast(
        self,
        lat: float,
        lon: float,
        target_time: datetime,
    ) -> WeatherData | None:
        """Fetch weather forecast for a specific location and time.

        Args:
            lat: Latitude
            lon: Longitude
            target_time: Match kickoff time (UTC)

        Returns:
            WeatherData or None if unavailable
        """
        date_str = target_time.strftime("%Y-%m-%d")
        hour = target_time.hour

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,cloud_cover,weather_code",
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "UTC",
        }

        try:
            response = httpx.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            hourly = data.get("hourly", {})
            temps = hourly.get("temperature_2m", [])
            humidity = hourly.get("relative_humidity_2m", [])
            wind = hourly.get("wind_speed_10m", [])
            precip = hourly.get("precipitation", [])
            clouds = hourly.get("cloud_cover", [])
            codes = hourly.get("weather_code", [])

            if not temps or hour >= len(temps):
                return None

            code = int(codes[hour]) if hour < len(codes) else 0

            return WeatherData(
                temperature_c=float(temps[hour]),
                humidity_pct=float(humidity[hour]) if hour < len(humidity) else 50.0,
                wind_speed_kmh=float(wind[hour]) if hour < len(wind) else 10.0,
                precipitation_mm=float(precip[hour]) if hour < len(precip) else 0.0,
                cloud_cover_pct=float(clouds[hour]) if hour < len(clouds) else 50.0,
                weather_code=code,
                description=WEATHER_CODES.get(code, "未知"),
            )
        except Exception as e:
            logger.warning(f"Weather fetch failed for ({lat}, {lon}): {e}")
            return None

    def fetch_for_venue(
        self,
        venue: str,
        target_time: datetime,
    ) -> WeatherData | None:
        """Fetch weather for a known venue."""
        coords = self._find_coords(venue)
        if coords is None:
            logger.debug(f"No coordinates for venue: {venue}")
            return None
        return self.fetch_forecast(coords[0], coords[1], target_time)

    def _find_coords(self, venue: str) -> tuple[float, float] | None:
        """Find coordinates for a venue string."""
        venue_lower = venue.lower()
        for name, coords in VENUE_COORDS.items():
            if name.lower() in venue_lower:
                return coords
        return None


def weather_to_features(weather: WeatherData | None) -> dict[str, float]:
    """Convert weather data to numeric features."""
    if weather is None:
        return {
            "weather_temp": 15.0,
            "weather_humidity": 50.0,
            "weather_wind": 10.0,
            "weather_precip": 0.0,
            "weather_clouds": 50.0,
            "weather_is_rain": 0.0,
            "weather_is_cold": 0.0,
            "weather_is_hot": 0.0,
            "weather_is_windy": 0.0,
        }

    return {
        "weather_temp": weather.temperature_c,
        "weather_humidity": weather.humidity_pct,
        "weather_wind": weather.wind_speed_kmh,
        "weather_precip": weather.precipitation_mm,
        "weather_clouds": weather.cloud_cover_pct,
        "weather_is_rain": 1.0 if weather.precipitation_mm > 0.5 else 0.0,
        "weather_is_cold": 1.0 if weather.temperature_c < 5 else 0.0,
        "weather_is_hot": 1.0 if weather.temperature_c > 30 else 0.0,
        "weather_is_windy": 1.0 if weather.wind_speed_kmh > 30 else 0.0,
    }
