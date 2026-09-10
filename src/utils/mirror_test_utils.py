"""Shared helpers for mirror_test sample gathering.

MIND-Data's mirror_test/ is FLAT:
  - 50 first-frame PNGs: data-25.png ... data-74.png (one per mem_test gt_name)
  - 10 directional "go-then-return" action JSONs: -w, -s, -a, -d, -u, -down, -wl, -wr, -sl, -sr
    each is a 47-frame trajectory in standard MIND action.json format.

The gsc metric scores by splitting the generated mp4 in half and time-flipping the
second half against the first. For comparable numbers across models, each driver
generates ONE canonical mirror mp4 per first-frame (default action prefix is "w"
= forward-out / back-return). Use --mirror-action to override per-run.

Sample dict returned by gather_mirror_samples mirrors the existing per-sample
shape used by the drivers, plus:
  - "video" is None (no source video — first frame is already a PNG)
  - "frame_png_src" is the path to the PNG (drivers should copy it instead of
    calling av-extract_first_frame).
"""

from pathlib import Path

PERSPECTIVES = ("1st_data", "3rd_data")
MIRROR_ACTIONS = ("w", "s", "a", "d", "u", "down", "wl", "wr", "sl", "sr")
MIRROR_DEFAULT_ACTION = "w"


def fit_mirror_frames(n_ticks: int, scale: int = 1, offset: int = 0) -> int:
    """Smallest valid clip length >= n_ticks for a `scale*k + offset` model.

    gsc splits the generated clip at its midpoint and expects the trajectory's
    turnaround to be there. The mirror trajectories are short -- ``-w.json`` is
    48 ticks, 24 out and 24 back -- so a driver that generates its usual clip
    length (h3world 124, echo 97, matrix-game-3 336) puts the turnaround in the
    first quarter and scores an overshoot rather than a return.

    Each model constrains its frame count differently, so callers pass their
    rule:

        echo (LTX, 8k+1)      fit_mirror_frames(n, 8, 1)   48 -> 49
        h3world (17k+5)       fit_mirror_frames(n, 17, 5)  48 -> 56
        unconstrained         fit_mirror_frames(n)         48 -> 48

    When the result exceeds n_ticks the trajectory should be RESAMPLED across
    it, not padded -- padding leaves the turnaround short of the midpoint.
    Echo's 49 is close enough to pad by one idle frame; h3world's 56 is not.
    """
    if n_ticks <= 0:
        raise ValueError(f"n_ticks must be positive, got {n_ticks}")
    if scale <= 1:
        return n_ticks
    return offset + scale * max(1, -(-(n_ticks - offset) // scale))


def gather_mirror_samples(gt_root: Path, action: str = MIRROR_DEFAULT_ACTION) -> list[dict]:
    """One sample per data-NN.png × <perspective>, paired with the chosen action JSON.

    Returns sample dicts compatible with each driver's existing per-sample loop:
        {perspective, test_type='mirror_test', gt_name, video=None, action, frame_png_src}
    Output path is <test_root>/<model>/<perspective>/mirror_test/<gt_name>/video.mp4.
    """
    if action not in MIRROR_ACTIONS:
        raise ValueError(f"action {action!r} not in {MIRROR_ACTIONS}")
    samples: list[dict] = []
    for perspective in PERSPECTIVES:
        mirror_dir = gt_root / perspective / "test" / "mirror_test"
        if not mirror_dir.is_dir():
            continue
        action_json = mirror_dir / f"-{action}.json"
        if not action_json.exists():
            continue
        pngs = sorted(
            p for p in mirror_dir.iterdir()
            if p.suffix.lower() == ".png" and p.stem.startswith("data-")
        )
        for png in pngs:
            samples.append({
                "perspective": perspective,
                "test_type": "mirror_test",
                "gt_name": png.stem,        # e.g. "data-25"
                "video": None,
                "action": action_json,
                "frame_png_src": png,
            })
    return samples
