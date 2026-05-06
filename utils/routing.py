"""Simple rule-based shipping route suggestions."""

ROUTE_TRIGGER_PHRASES = [
    "suggest route",
    "suggest a route",
    "suggest safer route",
    "suggest a safer route",
    "alternative route",
    "safer route",
    "reroute",
    "which route should i take",
]

ROUTES = [
    ["Dubai", "Hormuz", "Mumbai"],
    ["Dubai", "Salalah", "Mumbai"],
    ["Dubai", "Cape of Good Hope", "Mumbai"],
]

HIGH_RISK_AREAS = {"hormuz"}


def is_route_query(query: str) -> bool:
    """Check whether a user explicitly asks for a route suggestion."""
    clean_query = query.lower()
    clean_query = " ".join(clean_query.replace("?", " ").split())

    return any(phrase in clean_query for phrase in ROUTE_TRIGGER_PHRASES)


def route_has_high_risk_area(route: list[str]) -> bool:
    """Check if a route passes through a high-risk area."""
    route_stops = {stop.lower() for stop in route}
    return bool(route_stops & HIGH_RISK_AREAS)


def format_route(route: list[str]) -> str:
    """Convert a route list into readable text."""
    return " -> ".join(route)


def suggest_route(query: str) -> str:
    """Suggest a safer route using simple hardcoded maritime rules."""
    risky_routes = [route for route in ROUTES if route_has_high_risk_area(route)]
    safer_routes = [route for route in ROUTES if not route_has_high_risk_area(route)]

    if not safer_routes:
        return (
            "Suggested Route:\n"
            "No safer route is available.\n\n"
            "Reason:\n"
            "All known routes include high-risk areas."
        )

    # For this simple project, prefer the Salalah option because it avoids
    # Hormuz without adding as much delay as the Cape of Good Hope route.
    suggested_route = safer_routes[0]

    if "cape" in query.lower():
        suggested_route = safer_routes[-1]

    avoided_route = format_route(risky_routes[0]) if risky_routes else "the high-risk route"

    return (
        "Suggested Route:\n"
        f"{format_route(suggested_route)}\n\n"
        "Reason:\n"
        f"Avoid {avoided_route} because Hormuz is marked as high-risk. "
        "The suggested route is safer, but it may add some delay compared with "
        "the direct route."
    )


def suggest_alternative_route(origin: str, destination: str) -> str:
    """Return a route suggestion for the simple Streamlit form."""
    if not origin.strip() or not destination.strip():
        return "Please enter both an origin port and a destination port."

    query = f"Suggest a safer route from {origin} to {destination}"
    if origin.lower().strip() == "dubai" and destination.lower().strip() == "mumbai":
        return suggest_route(query)

    return (
        "Suggested Route:\n"
        f"{origin} -> {destination}\n\n"
        "Reason:\n"
        "No hardcoded alternative is available for this port pair yet."
    )
