"""Disposal and compliance pointers per part type.
A reference summary for sorting decisions. NOT legal advice: laws change, so check the
current official texts. Scope: India first, with pointers to EU and international rules."""

REVIEWED = "October 2026"
DISCLAIMER = ("Reference summary, not legal advice. Rules change, so check the current official texts "
              "(CPCB and the Ministry of Environment, Forest and Climate Change for India). "
              "Covers India, with pointers to EU and international rules. Last reviewed " + REVIEWED + ".")
IN_EW = "India: E-Waste (Management) Rules, 2022"
IN_BAT = "India: Battery Waste Management Rules, 2022"
EU = "EU: WEEE Directive and RoHS Directive"

RULES = {
    "battery": ("High", "May contain lead, cadmium, lithium or corrosive electrolyte. Fire risk if punctured or shorted.",
                [IN_BAT + " (waste batteries are outside the e-waste rules)", "EU: Batteries Regulation (EU) 2023/1542"],
                "Remove first, tape the terminals, keep dry, hand to an authorised battery recycler. Never crush or burn."),
    "display": ("High", "Older LCD backlights can contain mercury. Broken glass and chemicals.",
                [IN_EW, EU], "Do not break it. Send intact to a registered dismantler."),
    "capacitor": ("Medium", "Electrolytic types hold electrolyte and large ones can keep a charge. Very old or industrial capacitors can contain PCBs.",
                  [IN_EW, "Stockholm Convention (covers PCB-containing parts)"],
                  "Discharge large ones before handling, do not puncture, send to a registered recycler."),
    "ic": ("Medium", "Older boards often use leaded solder. Dust from burning or grinding boards is the main danger.",
           [IN_EW + " (restricts lead, mercury and cadmium in new equipment)", EU],
           "Desolder with ventilation. Do not burn boards. Scrap goes to a registered recycler."),
    "connector": ("Low", "Plated metals have recovery value. Plastic housings may contain flame retardants.",
                  [IN_EW], "Reuse if intact, otherwise send to a registered recycler."),
    "fuse": ("Low", "Older fuses may contain lead in solder or end caps.", [IN_EW], "Collect with other scrap for a registered recycler."),
}
RULES["microcontroller"] = RULES["ic"]
LOW = ("Low", "Small part. Solder on older boards may contain lead.", [IN_EW], "Reuse if good, otherwise include in scrap for a registered recycler.")
GENERAL = [
    "Under India's E-Waste (Management) Rules, 2022, dismantlers, refurbishers and recyclers must register with CPCB, and entities may not deal with unregistered ones. Sell or hand over scrap only to registered channels.",
    "Limits such as lead 0.1% and cadmium 0.01% apply to manufacturers of new equipment (RoHS-style rules). For sorting, the practical rule is to route hazardous parts to authorised recyclers.",
    "Moving hazardous e-waste across borders is controlled by the Basel Convention. Ask a specialist before exporting.",
]
ORDER = {"High": 3, "Medium": 2, "Low": 1}


def audit(classes):
    rows, worst = [], "Low"
    for c in classes:
        risk, why, rules, action = RULES.get(str(c).lower().strip(), LOW)
        rows.append({"class": str(c), "risk": risk, "why": why, "rules": rules, "action": action})
        if ORDER[risk] > ORDER[worst]:
            worst = risk
    rows.sort(key=lambda r: -ORDER[r["risk"]])
    return {"overall": worst, "rows": rows, "general": GENERAL, "disclaimer": DISCLAIMER}