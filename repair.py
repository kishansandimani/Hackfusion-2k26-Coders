"""Repair and replacement guides per part type, written in advance (not generated per board).
Specifications must be read from the part or its datasheet: a photo cannot show them."""

SAFE = "Switch off and unplug. Discharge large capacitors. Use an anti-static wrist strap and ventilation when soldering."
G = {
    "capacitor": ("Bulging top, leaking, brown residue, or the device resets or hums.",
                  "Same capacitance (about 20% either way is usually fine), same or higher voltage rating, same polarity, similar size. For power supplies choose a low-ESR type.",
                  ["Note the polarity stripe before removing.", "Desolder and clean the pads.", "Fit the new part with the stripe on the same side.", "Trim leads, check for solder bridges, power up carefully."]),
    "resistor": ("Burnt or discoloured body, or a measured value far from its marking.",
                 "Same resistance, equal or tighter tolerance, power rating at least the same, same package size.",
                 ["Measure it out of circuit with a multimeter.", "Read the colour or number code.", "Fit a matching part."]),
    "ic": ("Overheating, no output, or a visibly damaged package.",
           "The same part number, or an equivalent that the datasheet says is pin and voltage compatible.",
           ["Check the datasheet pinout and supply voltage.", "Use hot air or desoldering wick, with a socket if you can.", "Mind pin 1 and static electricity."]),
    "microcontroller": ("No response, or it does not run its program.",
                        "The same part number. Even close variants differ in pins, memory and boot code.",
                        ["Confirm the exact part on the chip.", "Replace, then reprogram it with a programmer or bootloader."]),
    "connector": ("Loose, bent or cracked pins, or an intermittent signal.",
                  "Same pitch, pin count, orientation and current rating.",
                  ["Compare with the old one on the board.", "Desolder pin by pin or with hot air.", "Support it mechanically."]),
    "led": ("Does not light, or lights dimly.", "Same colour, forward voltage and current, with the correct polarity.",
            ["Check polarity with a diode test.", "Keep the series resistor in place."]),
    "diode": ("Shorted or open in a diode test.", "Same type (rectifier, Schottky or Zener) with equal or higher voltage and current ratings.",
              ["The band marks the cathode.", "Fit the new one the same way round."]),
    "transistor": ("Shorted or open between its pins.", "The same type and pinout. Pin order differs between packages, so check the datasheet.",
                   ["Identify NPN, PNP or MOSFET.", "Match pin order to the datasheet before soldering."]),
    "inductor": ("Open coil or overheating.", "Same inductance, current rating and size.", ["Measure with an LCR meter if you have one."]),
    "fuse": ("Open circuit.", "Same current and voltage rating and the same speed type. Never fit a higher rating or a wire.", ["Find out why it blew before replacing it."]),
    "clock": ("No clock signal.", "Same frequency and load capacitance.", ["Check the markings on the crystal."]),
    "switch": ("Intermittent or dead.", "Same type and electrical rating.", ["Clean contacts first, replace if still faulty."]),
    "button": ("Intermittent or dead.", "Same size and travel.", ["Clean contacts first, replace if still faulty."]),
    "potentiometer": ("Scratchy or jumping values.", "Same resistance and the same taper (linear or logarithmic).", ["Clean with contact cleaner first."]),
    "battery": ("Swollen, leaking or will not hold charge.", "Do not repair. Replace with the same chemistry and voltage, and hand the old cell to an authorised recycler.", ["Never puncture, heat or short a cell."]),
    "display": ("Cracked, lines, or dead backlight.", "Usually not repairable at this level.", ["Send to a registered dismantler."]),
}
EQUIV = {
    "LM358": ("LM2904", "Similar dual op-amp. Check supply range and temperature grade."),
    "LM324": ("LM2902", "Similar quad op-amp. Check supply range."),
    "NE555": ("LM555 or TLC555", "Same pinout. TLC555 is a lower-power CMOS version."),
    "LM7805": ("L7805 or MC7805", "Same 5 V linear regulator."),
    "LM317": ("LM338", "Same adjustable-regulator family with higher current."),
    "ATMEGA328P": ("ATmega168", "Same pinout family but less memory. Check your program fits."),
    "L293D": ("SN754410", "Compatible pinout. Check current and voltage ratings."),
}


def guides_for(classes):
    out = []
    for c in classes:
        g = G.get(str(c).lower().strip())
        if g:
            out.append({"class": str(c), "when": g[0], "replace": g[1], "steps": g[2], "safety": SAFE})
    return out


def equiv_for(part):
    e = EQUIV.get(str(part).upper())
    return {"part": part, "equiv": e[0], "note": e[1] + " Always verify with the datasheet."} if e else None