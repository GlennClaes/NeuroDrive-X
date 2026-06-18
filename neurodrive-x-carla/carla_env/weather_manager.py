"""Weather preset and day-night management for CARLA."""

from __future__ import annotations

import logging
from typing import Any

try:
    import carla
except ImportError:  # pragma: no cover
    carla = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


class WeatherManager:
    """Apply weather presets, night mode, and rain mode to a CARLA world."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.current_weather = str(config.get("default", "ClearNoon"))

    def apply(self, world: Any, preset: str | None = None, night_mode: bool | None = None, rain_mode: bool | None = None) -> str:
        """Apply a named weather preset and return the effective preset name."""

        _require_carla()
        selected = preset or self.config.get("default", "ClearNoon")
        night = self.config.get("night_mode", False) if night_mode is None else night_mode
        rain = self.config.get("rain_mode", False) if rain_mode is None else rain_mode

        if night and rain:
            selected = "WetNight"
        elif night:
            selected = "ClearNight"
        elif rain and selected == "ClearNoon":
            selected = "WetCloudyNoon"

        weather = self._weather_for(selected)
        world.set_weather(weather)
        self.current_weather = selected
        LOGGER.info("Applied CARLA weather preset: %s", selected)
        return selected

    def _weather_for(self, preset: str) -> Any:
        _require_carla()
        built_in = getattr(carla.WeatherParameters, preset, None)
        if built_in is not None:
            return built_in

        custom = {
            "ClearNight": carla.WeatherParameters(
                cloudiness=10.0,
                precipitation=0.0,
                sun_altitude_angle=-25.0,
                fog_density=2.0,
            ),
            "WetNight": carla.WeatherParameters(
                cloudiness=75.0,
                precipitation=45.0,
                precipitation_deposits=80.0,
                wetness=90.0,
                sun_altitude_angle=-35.0,
                fog_density=18.0,
            ),
            "SoftRainSunset": carla.WeatherParameters(
                cloudiness=55.0,
                precipitation=25.0,
                precipitation_deposits=45.0,
                wetness=55.0,
                sun_altitude_angle=8.0,
            ),
        }
        if preset not in custom:
            LOGGER.warning("Unknown weather preset %s; falling back to ClearNoon.", preset)
        return custom.get(preset, carla.WeatherParameters.ClearNoon)


def _require_carla() -> None:
    if carla is None:
        raise RuntimeError("CARLA Python API is required for weather management.")

