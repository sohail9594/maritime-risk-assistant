"""Domain-specific decision rules layered on top of the RAG chatbot.

These rules handle important business edge cases where a simple retrieved
summary may miss the correct operational decision.
"""

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"

SENSITIVE_MEDICAL_TERMS = [
    "insulin",
    "vaccine",
    "vaccines",
    "medicine",
    "pharmaceutical",
    "pharmaceuticals",
    "surgical robotics",
    "medical equipment",
    "cold chain",
    "cold-chain",
]

COST_DIVERSION_TERMS = [
    "demurrage",
    "daily fee",
    "daily fees",
    "save money",
    "divert",
    "salalah",
    "truck overland",
    "truck it overland",
]

JEBEL_ALI_TERMS = ["jebel ali", "jebel"]
SALALAH_TERMS = ["salalah"]

SOURCE_KEYWORDS = {
    "jebel_fast_track": ["jebel ali", "fast-track", "12 to 24", "pharmaceuticals"],
    "salalah_cold_chain": ["salalah", "cold chain", "insulin", "shortage"],
    "demurrage": ["jebel ali", "demurrage", "$400"],
}

FRAGILE_ELECTRONICS_CARGO_TERMS = [
    "ai servers",
    "data center servers",
    "server racks",
    "racks",
    "gpu servers",
    "fragile electronics",
    "high-value electronics",
    "hardware",
]

RISKY_TRANSPORT_TERMS = [
    "unpaved road",
    "unpaved",
    "under-construction road",
    "under-construction",
    "under construction",
    "mountain pass",
    "mountain routes",
    "rough road",
    "overland trucks",
    "overland trucking",
    "vibration",
    "shock",
    "g-force",
    "g force",
    "tilt sensor",
    "tilt sensors",
    "shock sensor",
    "shock sensors",
]

IOT_CONTAINER_TERMS = [
    "iot",
    "telemetry",
    "smart container",
    "smart containers",
    "smart-container",
    "smart-containers",
    "tracking",
    "automated container",
    "automated containers",
]

TELEMETRY_DISRUPTION_TERMS = [
    "jamming",
    "signal blocked",
    "blocked signal",
    "blocking their signal",
    "no signal",
    "lost signal",
    "signal loss",
    "tracking not working",
    "connectivity disrupted",
    "communication disrupted",
]

BROAD_FAILURE_TERMS = [
    "not working",
    "offline",
    "broken",
]

SATELLITE_NAVIGATION_TERMS = [
    "satellite",
    "gps",
]


def clean_query(query: str) -> str:
    """Normalize a query for simple rule matching."""
    lowercase_query = query.lower()
    lowercase_query = re.sub(r"[^a-z0-9$\s-]", " ", lowercase_query)
    return " ".join(lowercase_query.split())


def contains_any(text: str, terms: list[str]) -> bool:
    """Return True if any term appears in the text."""
    return any(term in text for term in terms)


def is_fragile_electronics_transport_query(query: str) -> bool:
    """Detect fragile electronics moving through rough overland conditions.

    This is domain-specific rule-based decision logic layered on top of RAG.
    It runs before retrieval so AI server hardware is not confused with
    unrelated medical cold-chain cargo.
    """
    clean_text = clean_query(query)
    has_fragile_cargo = contains_any(clean_text, FRAGILE_ELECTRONICS_CARGO_TERMS)
    has_risky_transport = contains_any(clean_text, RISKY_TRANSPORT_TERMS)

    return has_fragile_cargo and has_risky_transport


def is_medical_diversion_decision_query(query: str) -> bool:
    """Detect cold-chain medical cargo diversion questions.

    This intentionally runs before normal RAG because the correct answer depends
    on a domain-specific decision rule, not just one retrieved paragraph.
    """
    clean_text = clean_query(query)
    has_sensitive_cargo = contains_any(clean_text, SENSITIVE_MEDICAL_TERMS)
    has_cost_or_diversion_signal = contains_any(clean_text, COST_DIVERSION_TERMS)
    mentions_jebel_ali = contains_any(clean_text, JEBEL_ALI_TERMS)
    mentions_salalah = contains_any(clean_text, SALALAH_TERMS)

    return (
        has_sensitive_cargo
        and has_cost_or_diversion_signal
        and mentions_jebel_ali
        and mentions_salalah
    )


def is_iot_satellite_jamming_query(query: str) -> bool:
    """Detect smart-container telemetry loss caused by GPS/satellite jamming.

    This is domain-specific rule-based decision logic layered on top of RAG.
    It separates telemetry problems from physical container failure.
    """
    clean_text = clean_query(query)
    has_iot_container_signal = contains_any(clean_text, IOT_CONTAINER_TERMS)
    has_telemetry_disruption = contains_any(clean_text, TELEMETRY_DISRUPTION_TERMS)
    has_broad_failure_question = contains_any(clean_text, BROAD_FAILURE_TERMS)
    mentions_satellite_navigation = contains_any(clean_text, SATELLITE_NAVIGATION_TERMS)
    mentions_tracking_or_telemetry = "tracking" in clean_text or "telemetry" in clean_text

    return (
        has_iot_container_signal
        and (
            has_telemetry_disruption
            or (
                (mentions_satellite_navigation or mentions_tracking_or_telemetry)
                and has_broad_failure_question
            )
        )
    )


def answer_iot_satellite_jamming_decision(query: str) -> str:
    """Return the special answer for IoT/satellite jamming scenarios."""
    return (
        "Risk Analysis:\n"
        "The containers are likely still functioning normally. Modern smart "
        "containers use onboard edge controllers to maintain temperature and "
        "operations locally, even if satellite communication is disrupted.\n\n"
        "Recommendation:\n"
        "The issue is limited to telemetry loss (data not reaching the dashboard), "
        "not physical failure. Once the containers leave the jamming zone, data "
        "transmission should resume. Continue monitoring locally and verify once "
        "connectivity is restored."
    )


def answer_fragile_electronics_decision(query: str) -> str:
    """Return the special decision answer for fragile AI server cargo."""
    return (
        "Decision:\n"
        "Do not use the unpaved mountain-pass truck route.\n\n"
        "Reasoning:\n"
        "Fragile AI/data-center servers are sensitive to vibration, shock, tilt, "
        "and G-force events. Rough overland transport can trigger shock "
        "indicators or tilt sensors, creating warranty and insurance problems "
        "even if the equipment appears externally undamaged.\n\n"
        "Operational Recommendation:\n"
        "Use a paved, controlled logistics corridor with air-ride suspension "
        "trucks, shock monitoring, and documented handling controls. Saving time "
        "is not worth the risk of damaging equipment or voiding manufacturer "
        "warranty."
    )


def find_decision_sources() -> list[str]:
    """Find source filenames that support the medical diversion decision."""
    if not DATA_FOLDER.exists():
        return []

    source_names = []

    for file_path in sorted(DATA_FOLDER.glob("*.txt")):
        try:
            text = file_path.read_text(encoding="utf-8").lower()
        except FileNotFoundError:
            continue

        for keywords in SOURCE_KEYWORDS.values():
            if all(keyword in text for keyword in keywords):
                if file_path.name not in source_names:
                    source_names.append(file_path.name)
                break

    return source_names


def answer_medical_diversion_decision(query: str) -> str:
    """Return the special decision answer for cold-chain medical cargo."""
    sources = find_decision_sources()

    if not sources:
        sources = ["No matching source found"]

    source_lines = "\n".join(f"- {source}" for source in sources[:3])

    return (
        "Decision:\n"
        "Stay with Jebel Ali. Do not divert to Salalah.\n\n"
        "Reasoning:\n"
        "Although demurrage fees are expensive, insulin qualifies for Fast-Track "
        "Priority clearance at Jebel Ali, reducing expected clearance time to "
        "approximately 12-24 hours. Diverting to Salalah may create additional "
        "cold-chain trucking delays and temperature-control risk.\n\n"
        "Cost/Risk Tradeoff:\n"
        "Paying short-term demurrage is safer than risking cold-chain disruption "
        "during overland trucking.\n\n"
        "Sources:\n"
        f"{source_lines}"
    )
