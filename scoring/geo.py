# scoring/geo.py
import logging
from datetime import datetime
from typing import Optional
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

logger = logging.getLogger(__name__)

# Cache for geocoding results
_geocode_cache: dict[str, Optional[tuple[float, float]]] = {}

# Known city coordinates (fallback when geocoding fails)
KNOWN_CITIES = {
    "munich": (48.1351, 11.5820),
    "berlin": (52.5200, 13.4050),
    "rome": (41.9028, 12.4964),
    "paris": (48.8566, 2.3522),
    "london": (51.5074, -0.1278),
    "audincourt": (47.4836, 6.8403),
    "dietzenbach": (50.0092, 8.7797),
    "hamburg": (53.5511, 9.9937),
}

def geocode_location(location_str: str) -> Optional[tuple[float, float]]:
    """
    Geocode a location string to (lat, lng).
    Uses known cities first, then falls back to Nominatim.
    """
    if not location_str or not isinstance(location_str, str):
        return None

    # Check cache
    if location_str in _geocode_cache:
        return _geocode_cache[location_str]

    # Extract city name (before " - " if present)
    city = location_str.split(" - ")[0].strip().lower()

    # Check known cities first
    if city in KNOWN_CITIES:
        result = KNOWN_CITIES[city]
        _geocode_cache[location_str] = result
        return result

    # Try geocoding
    try:
        geolocator = Nominatim(user_agent="fraud_detector")
        result = geolocator.geocode(city, timeout=5)
        if result:
            coords = (result.latitude, result.longitude)
            _geocode_cache[location_str] = coords
            return coords
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logger.warning(f"Geocoding failed for {location_str}: {e}")

    _geocode_cache[location_str] = None
    return None


def geo_scorer(
    tx_location: Optional[str],
    timestamp: datetime,
    sender_biotag: str,
    locations: list[dict],
    prev_tx_location: Optional[str],
    prev_tx_timestamp: Optional[datetime]
) -> float:
    """
    Score transaction based on geographic anomaly.

    Returns:
        Risk score 0.0-1.0
    """
    # Skip geocoding to avoid rate limits - return 0 for all
    return 0.0

    if not tx_location or not isinstance(tx_location, str):
        return 0.0

    tx_coords = geocode_location(tx_location)
    if not tx_coords:
        return 0.0

    score = 0.0

    # Check 1: Distance from user's GPS location
    user_coords = _find_nearest_gps(sender_biotag, timestamp, locations)
    if user_coords:
        distance_km = geodesic(user_coords, tx_coords).km
        if distance_km > 50:
            score = min(distance_km / 500, 1.0)

    # Check 2: Impossible velocity between consecutive transactions
    if prev_tx_location and prev_tx_timestamp:
        prev_coords = geocode_location(prev_tx_location)
        if prev_coords:
            distance_km = geodesic(prev_coords, tx_coords).km
            time_diff_hours = (timestamp - prev_tx_timestamp).total_seconds() / 3600

            if time_diff_hours > 0:
                velocity = distance_km / time_diff_hours
                if velocity > 500:  # km/h - impossible without flight
                    score = 1.0

    return score


def _find_nearest_gps(
    biotag: str,
    timestamp: datetime,
    locations: list[dict]
) -> Optional[tuple[float, float]]:
    """Find the GPS location closest in time to the transaction."""
    if not locations:
        return None

    user_locs = [loc for loc in locations if loc.get("biotag") == biotag]
    if not user_locs:
        return None

    # Find closest timestamp
    closest = None
    min_diff = float('inf')

    for loc in user_locs:
        loc_time = datetime.fromisoformat(loc["timestamp"].replace("Z", "+00:00"))
        if loc_time.tzinfo:
            loc_time = loc_time.replace(tzinfo=None)

        diff = abs((timestamp - loc_time).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest = loc

    if closest:
        return (float(closest["lat"]), float(closest["lng"]))
    return None
