"""Browser geolocation component for Streamlit."""
from pathlib import Path

import streamlit.components.v1 as components

_gps_component = components.declare_component(
    "food_finder_gps",
    path=str(Path(__file__).parent / "gps_component"),
)


def get_browser_location(key="browser_location"):
    return _gps_component(key=key, default=None)
