"""
Instruction-to-ARKit-blendshapes mapping for blendshapes review and stats.
Used to show histograms per instruction group.
"""
from typing import Dict, List, Tuple

ALL_BLENDSHAPES_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]

# All directional eye blendshapes (for "move your eyes in a circle")
ALL_DIRECTIONAL_EYE_BLENDSHAPES = [
    "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
]


def _parse_blendshape_list(s: str) -> List[str]:
    """Parse 'mouthSmileLeft, mouthSmileRight' into list of names. Drops parenthetical notes."""
    out = []
    for part in s.split(","):
        part = part.strip()
        if "(" in part:
            part = part.split("(")[0].strip()
        if part and part in ALL_BLENDSHAPES_NAMES:
            out.append(part)
    return out


# Each entry: (group_label, list of instruction line prefixes/strings that belong to this group, blendshape names)
# Instruction(s) column content split by newline gives multiple strings per row; all map to same blendshapes.
INSTRUCTION_GROUPS: List[Tuple[str, List[str], List[str]]] = [
    (
        "expression: smile",
        [
            "expression: smile",
            "expression: smile\nintensity: small",
            "expression: smile\nintensity: medium",
            "expression: smile\nintensity: big",
        ],
        _parse_blendshape_list("mouthSmileLeft, mouthSmileRight"),
    ),
    (
        "expression: puff cheeks",
        [
            "expression: puff cheeks",
            "expression: puff cheeks\nintensity: small",
            "expression: puff cheeks\nintensity: medium",
            "expression: puff cheeks\nintensity: big",
        ],
        _parse_blendshape_list("cheekPuff"),
    ),
    (
        "expression: raise eyebrows",
        [
            "expression: raise eyebrows",
            "expression: raise eyebrows\nintensity: small",
            "expression: raise eyebrows\nintensity: medium",
            "expression: raise eyebrows\nintensity: big",
        ],
        _parse_blendshape_list("browInnerUp, browOuterUpLeft, browOuterUpRight"),
    ),
    (
        "expression: open mouth",
        [
            "expression: open mouth",
            "expression: open mouth\nintensity: small",
            "expression: open mouth\nintensity: medium",
            "expression: open mouth\nintensity: big",
        ],
        _parse_blendshape_list("jawOpen"),
    ),
    ("expression: close eyes", ["expression: close eyes"], _parse_blendshape_list("eyeBlinkLeft, eyeBlinkRight")),
    ("expression: look up", ["expression: look up"], _parse_blendshape_list("eyeLookUpLeft, eyeLookUpRight")),
    ("expression: look down", ["expression: look down"], _parse_blendshape_list("eyeLookDownLeft, eyeLookDownRight")),
    (
        "expression: look left",
        ["expression: look left"],
        _parse_blendshape_list("eyeLookOutLeft, eyeLookInRight"),
    ),
    (
        "expression: look right",
        ["expression: look right"],
        _parse_blendshape_list("eyeLookInLeft, eyeLookOutRight"),
    ),
    (
        "look up, then down",
        [
            "look up, then down (1 time)",
            "look up, then down (2 times)",
            "look up, then down (3 times)",
            "look up, then down (4 times)",
        ],
        _parse_blendshape_list("eyeLookUpLeft, eyeLookUpRight, eyeLookDownLeft, eyeLookDownRight"),
    ),
    (
        "look left, then right",
        [
            "look left, then right (1 time)",
            "look left, then right (2 times)",
            "look left, then right (3 times)",
            "look left, then right (4 times)",
        ],
        _parse_blendshape_list("eyeLookOutLeft, eyeLookInRight, eyeLookInLeft, eyeLookOutRight"),
    ),
    (
        "look from top left to bottom right",
        [
            "look from top left to bottom right (1 time)",
            "look from top left to bottom right (2 times)",
            "look from top left to bottom right (3 times)",
            "look from top left to bottom right (4 times)",
        ],
        _parse_blendshape_list(
            "eyeLookUpLeft, eyeLookUpRight, eyeLookOutLeft, eyeLookInRight, eyeLookDownLeft, eyeLookDownRight, eyeLookInLeft, eyeLookOutRight"
        ),
    ),
    (
        "look from top right to bottom left",
        [
            "look from top right to bottom left (1 time)",
            "look from top right to bottom left (2 times)",
            "look from top right to bottom left (3 times)",
            "look from top right to bottom left (4 times)",
        ],
        _parse_blendshape_list(
            "eyeLookUpLeft, eyeLookUpRight, eyeLookInLeft, eyeLookOutRight, eyeLookDownLeft, eyeLookDownRight, eyeLookOutLeft, eyeLookInRight"
        ),
    ),
    (
        "move your eyes in a circle",
        [
            "move your eyes in a circle (1 time)",
            "move your eyes in a circle (2 times)",
            "move your eyes in a circle (3 times)",
            "move your eyes in a circle (4 times)",
        ],
        ALL_DIRECTIONAL_EYE_BLENDSHAPES,
    ),
    (
        "ba ba ba. ma ma ma. pa pa pa.",
        [],
        _parse_blendshape_list("mouthClose, jawOpen, mouthPressLeft, mouthPressRight"),
    ),
    (
        "cha cha cha. fa fa fa. da da da.",
        [],
        _parse_blendshape_list("jawOpen, mouthRollLower, mouthUpperUpLeft, mouthUpperUpRight"),
    ),
    (
        "eee eye eee eye ohh",
        [],
        _parse_blendshape_list(
            "jawOpen, mouthSmileLeft, mouthSmileRight, mouthStretchLeft, mouthStretchRight, mouthFunnel, mouthPucker"
        ),
    ),
    ("cow. john. moose.", [], _parse_blendshape_list("jawOpen, mouthFunnel, mouthPucker")),
    ("one one one one one", [], _parse_blendshape_list("mouthPucker, jawOpen, mouthClose")),
]

# Add string variants for speech rows (read_text may be exactly the group label)
for i, (label, strings, blends) in enumerate(INSTRUCTION_GROUPS):
    if not strings and label:
        INSTRUCTION_GROUPS[i] = (label, [label], blends)

# Build: instruction_string -> group_label (for matching read_text)
READ_TEXT_TO_GROUP: Dict[str, str] = {}
GROUP_TO_BLENDSHAPES: Dict[str, List[str]] = {}
for group_label, instruction_strings, blendshape_names in INSTRUCTION_GROUPS:
    GROUP_TO_BLENDSHAPES[group_label] = blendshape_names
    for s in instruction_strings:
        READ_TEXT_TO_GROUP[s.strip()] = group_label


def get_group_for_read_text(read_text: str) -> str | None:
    """Return instruction group label if read_text matches one of the instruction strings, else None."""
    t = read_text.strip() if isinstance(read_text, str) else ""
    if t in READ_TEXT_TO_GROUP:
        return READ_TEXT_TO_GROUP[t]
    # Try startswith for longer instructions
    for s, group in sorted(READ_TEXT_TO_GROUP.items(), key=lambda x: -len(x[0])):
        if t == s or t.startswith(s) or s.startswith(t):
            return group
    return None


def get_blendshape_indices(names: List[str]) -> List[int]:
    """Return indices into ALL_BLENDSHAPES_NAMES for the given blendshape names (npz columns assumed same order)."""
    name_to_idx = {n: i for i, n in enumerate(ALL_BLENDSHAPES_NAMES)}
    return [name_to_idx[n] for n in names if n in name_to_idx]
