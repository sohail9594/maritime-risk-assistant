"""Generate mock maritime text documents for RAG testing."""

from pathlib import Path


MOCK_DOCUMENTS = {
    "port_congestion_singapore.txt": (
        "The Port of Singapore is experiencing moderate congestion due to a rise "
        "in container arrivals from East Asia. Several vessels have reported "
        "berthing delays of 18 to 30 hours during peak arrival windows. Terminal "
        "operators are prioritizing refrigerated cargo, medical supplies, and "
        "time-sensitive transshipment containers. Shipping lines are advised to "
        "confirm berth availability before final approach."
    ),
    "strait_of_hormuz_risk_advisory.txt": (
        "Vessels transiting near the Strait of Hormuz should maintain heightened "
        "situational awareness due to regional security concerns. Operators are "
        "advised to review voyage plans, update emergency contact lists, and keep "
        "automatic identification systems active unless instructed otherwise by "
        "authorities. Tankers carrying energy cargo may face additional insurance "
        "checks before entering the area. Masters should report suspicious activity "
        "to the nearest maritime security center."
    ),
    "red_sea_shipping_delay_notice.txt": (
        "Shipping schedules through the Red Sea corridor remain vulnerable to "
        "security-related disruptions and convoy timing changes. Some carriers "
        "have adjusted sailing windows to reduce exposure during high-risk periods. "
        "Cargo owners should expect possible delays of three to seven days on "
        "routes connecting Asia, the Middle East, and Europe. Freight forwarders "
        "are encouraged to update customers with revised estimated arrival dates. "
        "Alternative routings may increase transit time but can improve operational "
        "certainty."
    ),
    "cape_of_good_hope_alternative_route.txt": (
        "Several shipping companies are evaluating the Cape of Good Hope as an "
        "alternative route for Asia-Europe voyages. This route avoids some security "
        "risks in narrow waterways but adds distance, fuel consumption, and crew "
        "planning requirements. Longer transit times may affect inventory levels "
        "for manufacturers using just-in-time supply chains. Charterers should "
        "compare bunker costs and schedule reliability before approving route "
        "changes. The route may be suitable for lower-priority cargo where safety "
        "is more important than speed."
    ),
    "marine_insurance_cost_update.txt": (
        "Marine insurers are reviewing war-risk premiums for vessels operating near "
        "politically sensitive chokepoints. Premiums may increase for tankers, bulk "
        "carriers, and container ships that pass through areas with recent security "
        "incidents. Shipowners should contact their insurance providers before "
        "confirming final voyage instructions. Higher insurance costs may be passed "
        "on to customers through freight surcharges. Companies with flexible supply "
        "chains may consider safer but longer routes to control overall risk."
    ),
    "weather_delay_indian_ocean.txt": (
        "Weather conditions in parts of the Indian Ocean are causing slower vessel "
        "speeds and minor schedule disruptions. Masters have reported rough seas, "
        "strong winds, and reduced visibility along several eastbound lanes. "
        "Container lines may adjust arrival estimates to protect crew safety and "
        "reduce cargo damage. Port agents should prepare for bunching when delayed "
        "vessels arrive close together. Shippers are advised to monitor carrier "
        "notices for updated arrival windows."
    ),
    "port_of_rotterdam_capacity_notice.txt": (
        "The Port of Rotterdam has issued a capacity notice following increased "
        "container yard utilization. Import containers may require faster pickup to "
        "avoid storage pressure at inland connections. Rail and barge operators are "
        "coordinating extra slots to reduce gate congestion. Exporters should "
        "deliver cargo within assigned time windows to prevent missed vessel "
        "connections. Terminal planners expect conditions to improve once delayed "
        "feeder services are cleared."
    ),
    "suez_canal_transit_planning.txt": (
        "Shipping companies planning Suez Canal transits should verify convoy "
        "schedules and documentation requirements before arrival. Late paperwork "
        "can result in missed convoy slots and additional anchorage time. Vessels "
        "with tight delivery commitments should build extra buffer time into their "
        "voyage plans. Canal agents can help coordinate inspection, toll payment, "
        "and pilotage arrangements. Route planners should compare canal transit "
        "time with longer alternatives when risk levels change."
    ),
}


def create_mock_data() -> None:
    """Create the data folder and write mock maritime documents."""
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)

    for file_name, content in MOCK_DOCUMENTS.items():
        file_path = data_folder / file_name
        file_path.write_text(content + "\n", encoding="utf-8")

    print(f"Created {len(MOCK_DOCUMENTS)} mock maritime documents in '{data_folder}'.")


if __name__ == "__main__":
    create_mock_data()
