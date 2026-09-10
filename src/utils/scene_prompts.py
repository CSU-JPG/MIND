"""Scene prompts for the MIND drivers.

MIND's action.json carries no caption field, so every driver has to supply its
own text. Two sets live here:

**Baseline** -- the short perspective-flavoured lines the drivers have always
used. MIND scores only compare across models when every run uses identical
text, so these must not change: altering them silently invalidates comparisons
against every previously scored model.

**Enhanced** -- opt-in via each driver's ``--enhance-prompt``. Longer, and aimed
at a specific observed failure rather than at sounding better: with only a
generic caption the model has nothing to hold onto, and a detail-heavy sample
was seen drifting to unrelated content by frame 30 (see drive_evoke.py's
--image-noise-sigma-min, where the same problem was attacked from the other side
by anchoring harder to the seed image). These name the things that should stay
fixed -- lighting, materials, palette, architecture -- and state that no new
location or time of day should appear.

Kept apart from the drivers so prompt wording can be iterated without touching
driver logic, and so every driver adopting --enhance-prompt shares one text
rather than drifting into per-driver variants.
"""

from __future__ import annotations

FIRST_PERSON = "1st_data"

BASELINE_PROMPTS = {
    "1st_data": "First-person view exploring a 3D virtual environment.",
    "3rd_data": "Third-person view of a character exploring a 3D virtual environment.",
}

ENHANCED_PROMPTS = {
    "1st_data": (
        "First-person view exploring a 3D virtual environment. The camera moves "
        "smoothly through the space already established in the opening frame, "
        "holding its lighting, materials, colour palette and architecture "
        "unchanged throughout. Surfaces keep their texture and geometry as the "
        "viewpoint moves; no new locations, weather or times of day appear."
    ),
    "3rd_data": (
        "Third-person view of a character exploring a 3D virtual environment. "
        "The camera follows the character through the space already established "
        "in the opening frame, holding its lighting, materials, colour palette "
        "and architecture unchanged throughout. The character keeps a consistent "
        "appearance and proportions; no new locations, weather or times of day "
        "appear."
    ),
}


def scene_prompt(perspective: str, enhance: bool) -> str:
    """The prompt for a perspective, baseline unless ``enhance`` is set.

    Anything that is not the first-person key is treated as third-person, which
    is what the drivers did when this was an if/else on ``== "1st_data"``.
    """
    key = FIRST_PERSON if perspective == FIRST_PERSON else "3rd_data"
    return (ENHANCED_PROMPTS if enhance else BASELINE_PROMPTS)[key]
