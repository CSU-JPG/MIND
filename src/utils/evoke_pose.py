"""Convert a MIND action.json trajectory into an Evoke-compatible vipe-style cam_c2w array.

Unlike the ws/ad/ud/lr tri-state key sequences some other drivers convert (dead-reckoning),
MIND's action.json also carries real per-tick ground-truth pose: actor_pos/actor_rpy (both
perspectives) and camera_pos/camera_rpy (3rd_data only -- the actual third-person camera,
offset from the followed actor). We use this directly instead of dead-reckoning from ws/ad/ud/lr,
since it's ground truth rather than an approximation.

Axis convention: MIND positions are UE-style (X fwd, Y right, Z up, cm, left-handed) with rpy =
{x: roll (always 0 in observed data), y: pitch, z: yaw, degrees}. Evoke's cam_c2w (see
examples/racer/build_pose_npz.py and examples/2/build_pose_npz.py in the Evoke repo) is
right-handed Y-up, forward = local -Z, yaw = rotation about Y. There is no documented exact
mapping between the two systems, so this is a best-effort conversion (position scale is
arbitrary -- vipe reconstructions have arbitrary scale too, so this matches that precedent),
not a verified-correct one. Roll/pitch are dropped (matches the racer/2 converters, which also
only model yaw); only positions and yaw are converted.

Verified empirically (2026-08-17): action.json tick count == video.mp4 frame count, both at
24fps, across every sample checked -- so no fps resampling is needed, ticks map 1:1 to frames.

pos_scale calibration (2026-08-17): the first MIND-driven sample (data-1-1.0x-200) came out as
an unrelated grid/mesh artifact instead of the GT scene -- a known failure signature of Evoke's
warp/geometric-conditioning system when fed a near-degenerate (too-small) camera trajectory.
Measured real per-frame UE translation for that sample: median ~5.3 cm/tick when moving. Real
vipe pose tracks (examples/i2v/pose.npz) move ~0.001-0.01 units/frame. The original 1/50000 scale
put our trajectory at ~1e-4 units/frame -- 10-100x smaller than real vipe magnitude, i.e. the
camera looked nearly stationary to the warp system. 1/1000 puts the same 5.3cm delta at ~0.0053,
in-range. Still a rough calibration (one sample, no verified ground truth for "correct" scale),
not a proven-correct value.
"""
import numpy as np


def mind_action_to_c2w(action_json: dict, perspective: str, pos_scale: float = 1.0 / 1000.0) -> np.ndarray:
    """Return cam_c2w [N,4,4] float32, one entry per action.json tick (== one per video frame).

    perspective: "1st_data" uses actor_pos/actor_rpy (the actor's own eye is the camera).
    "3rd_data" prefers camera_pos/camera_rpy (the actual filmed third-person viewpoint) when
    present, falling back to actor_pos/actor_rpy for older/partial samples.
    """
    frames = action_json["data"]
    use_camera = perspective == "3rd_data" and "camera_pos" in frames[0]
    pos_key, rpy_key = ("camera_pos", "camera_rpy") if use_camera else ("actor_pos", "actor_rpy")

    pos0 = frames[0][pos_key]
    yaw0 = frames[0][rpy_key]["z"]

    c2ws = np.zeros((len(frames), 4, 4), dtype=np.float32)
    for i, f in enumerate(frames):
        p, rpy = f[pos_key], f[rpy_key]
        # UE (X fwd, Y right, Z up) -> Evoke (X, Y up, Z) with forward = -Z convention.
        dx = (p["x"] - pos0["x"]) * pos_scale
        dy = (p["z"] - pos0["z"]) * pos_scale       # UE Z (up) -> Evoke Y (up)
        dz = -(p["y"] - pos0["y"]) * pos_scale      # UE Y (right) -> Evoke -Z

        yaw = np.deg2rad(rpy["z"] - yaw0)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [cos_y, 0.0, sin_y],
            [0.0, 1.0, 0.0],
            [-sin_y, 0.0, cos_y],
        ], dtype=np.float32)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = [dx, dy, dz]
        c2ws[i] = c2w

    return c2ws


def default_intrinsic() -> np.ndarray:
    """Normalized default intrinsic (auto-rescaled to source_resolution by load_pose_for_v2v)."""
    return np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
