"""Simple trucking time estimates for logistics questions."""

import re
from pathlib import Path
from typing import Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"

TRUCKING_TIME_TRIGGERS = [
    "how long",
    "time",
    "hours",
    "by truck",
    "trucking",
    "reach dubai",
]

TRUCKING_DURATIONS = {
    ("salalah", "dubai"): ("Salalah", "Dubai", "18-22 hours"),
    ("dubai", "abu dhabi"): ("Dubai", "Abu Dhabi", "2 hours"),
    ("dubai", "muscat"): ("Dubai", "Muscat", "5-6 hours"),
    ("salalah", "muscat"): ("Salalah", "Muscat", "10-12 hours"),
}


def clean_query(query: str) -> str:
    """Normalize a query for simple keyword checks."""
    lowercase_query = query.lower()
    lowercase_query = re.sub(r"[^a-z0-9\s]", " ", lowercase_query)
    return " ".join(lowercase_query.split())


def find_route_in_query(query: str) -> Tuple[str, str]:
    """Find a known origin and destination pair mentioned in the query."""
    clean_text = clean_query(query)

    for origin, destination in TRUCKING_DURATIONS:
        if origin in clean_text and destination in clean_text:
            return origin, destination

    return "", ""


def is_trucking_time_query(query: str) -> bool:
    """Check if a user is asking for a trucking duration estimate."""
    clean_text = clean_query(query)
    has_time_trigger = any(trigger in clean_text for trigger in TRUCKING_TIME_TRIGGERS)
    origin, destination = find_route_in_query(query)

    return has_time_trigger and bool(origin and destination)


def find_logistics_source(origin: str, destination: str) -> Optional[str]:
    """Find a relevant source file in the data folder for the route."""
    if not DATA_FOLDER.exists():
        return None

    best_source = None
    best_score = 0

    for file_path in sorted(DATA_FOLDER.glob("*.txt")):
        try:
            text = file_path.read_text(encoding="utf-8").lower()
        except FileNotFoundError:
            continue

        score = 0

        if origin.lower() in text:
            score += 2
        if destination.lower() in text:
            score += 2
        if "trucking" in text or "overland" in text:
            score += 2
        if "transit" in text:
            score += 1
        if "operational" in text:
            score += 5

        if score > best_score:
            best_source = file_path.name
            best_score = score

    if best_score < 4:
        return None

    return best_source


def estimate_trucking_time(origin: str, destination: str) -> str:
    """Return a simple hardcoded trucking time estimate."""
    route_key = (origin.lower().strip(), destination.lower().strip())
    route_info = TRUCKING_DURATIONS.get(route_key)

    if route_info is None:
        return (
            "Estimated Trucking Time:\n"
            f"{origin} -> {destination}: no mock estimate is available.\n\n"
            "Operational Note:\n"
            "No matching trucking estimate is available in the hardcoded table.\n\n"
            "Sources:\n"
            "- No matching source found"
        )

    display_origin, display_destination, duration = route_info
    source = find_logistics_source(display_origin, display_destination)

    if source:
        source_line = f"- {source}"
        operational_note = (
            f"The retrieved logistics bulletin says the {display_origin} to "
            f"{display_destination} trucking corridor is operational."
        )
    else:
        source_line = "- No matching source found"
        operational_note = (
            "No matching logistics bulletin was found; this estimate comes from "
            "the hardcoded mock duration table."
        )

    return (
        "Estimated Trucking Time:\n"
        f"{display_origin} -> {display_destination}: approximately {duration} by truck.\n\n"
        "Operational Note:\n"
        f"{operational_note}\n\n"
        "Sources:\n"
        f"{source_line}"
    )


def estimate_trucking_time_from_query(query: str) -> str:
    """Extract a route from the user query and return a trucking estimate."""
    origin, destination = find_route_in_query(query)

    if not origin or not destination:
        return (
            "Estimated Trucking Time:\n"
            "Please include a supported origin and destination.\n\n"
            "Operational Note:\n"
            "Supported mock routes are Salalah-Dubai, Dubai-Abu Dhabi, "
            "Dubai-Muscat, and Salalah-Muscat.\n\n"
            "Sources:\n"
            "- No matching source found"
        )

    return estimate_trucking_time(origin, destination)
