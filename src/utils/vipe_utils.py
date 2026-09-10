import os, sys
print(f"[VIPE_UTILS_DEBUG_v3] loading from {__file__} | pid={os.getpid()} | sys.path[0]={sys.path[0] if sys.path else '?'}", file=sys.stderr, flush=True)
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import shutil
import subprocess


def _resolve_vipe_cli() -> str:
    """Locate the ViPE CLI. Prefer the running interpreter's own venv, then
    known checkout locations, and only fall back to PATH.

    ViPE is installed editable into the MIND venv (built against torch 2.10
    + cu128 + sm_120 — see build_vipe.bat). The DeepVerse copy is kept as a
    fallback in case MIND's build is missing / broken.

    PATH used to be checked FIRST. On Linux that is actively wrong: moreutils
    ships an unrelated `/usr/bin/vipe` ("edit pipe in editor") which accepts
    none of ViPE's flags, so every sample died with

        Unknown option: output / Unknown option: pipeline
        Usage: /usr/bin/vipe [--suffix=extension]

    The action metric catches that per-sample and continues, so the run
    completed and simply reported no `action` at all — 136 of 137 samples
    skipped in one zing run. It only worked when the MIND venv happened to be
    activated, which put the right `vipe` earlier on PATH; running the same
    command from a non-activated shell (e.g. under tmux) silently lost the
    metric.

    The venv `bin/` (or `Scripts/`) beside `sys.executable` is the ViPE that
    belongs to this interpreter, so resolve that first and platform-agnostically
    — the old fallbacks were all Windows `Scripts/vipe.exe` paths and could
    never match on Linux.

    Function-scope imports guard against multiprocessing-spawn issues where a
    worker re-imports a stale or partial vipe_utils module: even if module-level
    state is weird, `os` and `shutil` are imported fresh here every call."""
    import os as _os
    import shutil as _shutil
    import sys as _sys

    _bindir = Path(_sys.executable).parent
    _siblings = Path(__file__).resolve().parent.parent.parent.parent
    candidates = [
        # This interpreter's own venv -- the ViPE built against the torch that
        # is already loaded in-process.
        _bindir / "vipe",
        _bindir / "vipe.exe",
        # MIND venv: torch 2.10 has sm_120 SASS so UniDepth's F.linear /
        # camera_layer no longer fails on RTX 5090.
        _siblings / "MIND" / ".venv" / "bin" / "vipe",
        _siblings / "MIND" / ".venv" / "Scripts" / "vipe.exe",
        # DeepVerse venv (fallback): torch 2.7, action metric fails on sm_120.
        _siblings / "DeepVerse" / ".venv" / "bin" / "vipe",
        _siblings / "DeepVerse" / ".venv" / "Scripts" / "vipe.exe",
    ]
    for c in candidates:
        if _os.path.exists(c):
            return str(c)
    found = _shutil.which("vipe")
    if found:
        return found
    return "vipe"  # let subprocess surface the original error


_VIPE_CLI = _resolve_vipe_cli()
import numpy as np
import os
from tqdm import tqdm

# -----------------------------
# SE(3) helpers
# -----------------------------

def quat_to_R_wxyz(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q /= n
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

def make_T(R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_wc
    T[:3, 3] = t_wc
    return T

def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def rot_angle_deg(R: np.ndarray) -> float:
    tr = np.trace(R)
    cos = (tr - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))

# -----------------------------
# Sim(3) Umeyama alignment
# -----------------------------

def umeyama_sim3(X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate similarity transform (s, R, t) such that:
        Y ≈ s * R * X + t
    X, Y: (N,3)
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    N = X.shape[0]
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY
    varX = (Xc * Xc).sum() / N
    Sigma = (Yc.T @ Xc) / N
    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (varX + 1e-12))
    t = muY - s * (R @ muX)
    return s, R, t

def apply_sim3_to_poses(Ts: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Apply Sim(3) to camera->world SE(3) poses:
      p' = s R p + t
      R' = R R_pose
    """
    out = Ts.copy()
    out[:, :3, :3] = R @ out[:, :3, :3]
    out[:, :3, 3] = (s * (R @ out[:, :3, 3].T)).T + t[None, :]
    return out

# -----------------------------
# RPE
# -----------------------------

def compute_rpe(gt_T: np.ndarray, est_T: np.ndarray, delta: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Relative Pose Error per step:
      dQ = Q_i^{-1} Q_{i+Δ}
      dP = P_i^{-1} P_{i+Δ}
      E  = dQ^{-1} dP
    Return arrays:
      trans_err: ||t(E)||
      rot_err_deg: angle(R(E))
    """
    n = min(len(gt_T), len(est_T))
    gt_T = gt_T[:n]
    est_T = est_T[:n]
    m = n - delta
    trans = np.zeros((m,), dtype=np.float64)
    rotdeg = np.zeros((m,), dtype=np.float64)
    for i in range(m):
        dQ = inv_T(gt_T[i]) @ gt_T[i + delta]
        dP = inv_T(est_T[i]) @ est_T[i + delta]
        E = inv_T(dQ) @ dP
        trans[i] = np.linalg.norm(E[:3, 3])
        rotdeg[i] = rot_angle_deg(E[:3, :3])
    return trans, rotdeg

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, quiet: bool = False, env: Optional[Dict[str, str]] = None) -> None:
    """
    Run a shell command.

    Args:
        cmd: Command and arguments as a list
        cwd: Working directory to run command in
        quiet: If True, suppress stdout output (stderr is always captured)
        env: Environment variables to set (e.g., CUDA_VISIBLE_DEVICES)
    """
    if not quiet:
        tqdm.write(f"Running command: {' '.join(cmd)}")

    # Merge with existing environment, then STRIP cross-venv pollution.
    # Defensive even now that vipe.exe lives in MIND venv (same python as
    # process.py): if a user runs process.py from a different venv (e.g. uv-managed
    # outer env with PYTHONHOME set), MIND's vipe.exe needs a clean env to load
    # its own python correctly. Was originally added for the DeepVerse-vipe path
    # (cp310 calling cp312) and still protects the MIND-vipe path.
    proc_env = os.environ.copy()
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONNOUSERSITE",
              "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT"):
        proc_env.pop(k, None)
    if env:
        proc_env.update(env)

    # text=True alone uses the OS locale codec (cp1252 on Windows), and ViPE's
    # stderr contains tqdm bars / unicode arrows that are not cp1252 — that
    # raises UnicodeDecodeError(`charmap`) inside run_cmd before we even see
    # the real return code. Force utf-8 with replacement.
    if quiet:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace", env=proc_env)
    else:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                          stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace", env=proc_env)

    if p.returncode != 0:
        error_msg = f"Command failed: {' '.join(cmd)}"
        if p.stderr:
            error_msg += f"\nError output:\n{p.stderr}"
        raise RuntimeError(error_msg)


def find_images_txt(root: Path) -> Optional[Path]:
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn == "images.txt":
                return (Path(dp) / fn)


def vipe_to_colmap(out_dir: Path, vipe_repo: Path, gpu_id: int = 0) -> None:
    vipe_repo = vipe_repo.resolve()
    script = Path("scripts/vipe_to_colmap.py")
    # sys.executable instead of bare "python": the action-metric worker runs in
    # MIND venv (cv2 4.13.0 present); bare `python` resolves via PATH to the
    # uv-managed system python which has no cv2 -> ModuleNotFoundError.
    run_cmd([sys.executable, str(script), str(out_dir)], cwd=vipe_repo, quiet=True, env={"CUDA_VISIBLE_DEVICES": str(gpu_id)})


def parse_colmap_images_txt(images_txt: Path) -> Tuple[np.ndarray, int]:
    """
    Parse COLMAP images.txt -> camera->world poses (T_wc) (N,4,4).
    COLMAP stores world->cam: qvec(w,x,y,z), tvec.
    Convert:
      R_wc = R_cw^T
      C_w  = -R_wc t_cw

    Returns:
        poses: (N, 4, 4) array of camera poses
        num_images: number of images parsed
    """
    lines = images_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    entries: List[Tuple[str, np.ndarray]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        # skip points2D line
        if i < len(lines):
            i += 1

        R_cw = quat_to_R_wxyz(qw, qx, qy, qz)
        t_cw = np.array([tx, ty, tz], dtype=np.float64)
        R_wc = R_cw.T
        C_w = -R_wc @ t_cw
        entries.append((name, make_T(R_wc, C_w)))

    if not entries:
        raise RuntimeError(f"No valid entries in {images_txt}")

    entries.sort(key=lambda x: x[0])
    poses = np.stack([e[1] for e in entries], axis=0)
    return poses, len(entries)

def extract_traj(video: Path, cache_dir: Path, pipeline: str = "default", gt_cache_path: Path = None,
                 expected_frames: int = 97, verbose_prefix: str = "", gpu_id: int = 0) -> np.ndarray:
    """
    Extract camera trajectory from video using ViPE.

    Args:
        video: Path to video file
        cache_dir: Directory for caching ViPE outputs
        pipeline: ViPE pipeline to use
        gt_cache_path: Optional path to cache/load images.txt for GT videos (e.g., video directory)
        expected_frames: Expected number of frames, used to validate cache
        verbose_prefix: 日志前缀
        gpu_id: GPU ID to use (default: 0)

    Returns:
        Array of camera poses (N, 4, 4)
    """
    video = video.resolve()
    cache_dir = cache_dir.resolve()

    # Check if GT cache exists
    if gt_cache_path is not None:
        gt_cache_path = Path(gt_cache_path).resolve()
        cached_images_txt = gt_cache_path / "images.txt"

        if cached_images_txt.exists():
            poses, num_images = parse_colmap_images_txt(cached_images_txt)
            if num_images < expected_frames:
                tqdm.write(f"{verbose_prefix}  Cache validation failed: expected {expected_frames} frames, got {num_images}. Re-running ViPE...")
            else:
                tqdm.write(f"{verbose_prefix}  Using cached GT trajectory: {cached_images_txt.name}, num_images:{num_images}, expected:{expected_frames}")
                return poses[:expected_frames]

    # Run ViPE inference (quietly)
    key = f"{video.name}-{_sha1(str(video))}"
    out_dir = cache_dir / "vipe" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"{verbose_prefix}  Running ViPE inference: {video.name} -> {out_dir.name}...")
    run_cmd([_VIPE_CLI, "infer", str(video), "--output", str(out_dir), "--pipeline", pipeline],
            quiet=True, env={"CUDA_VISIBLE_DEVICES": str(gpu_id)})

    tqdm.write(f"{verbose_prefix}  Converting to COLMAP format...")
    vipe_to_colmap(out_dir, Path("vipe"), gpu_id=gpu_id)
    images_txt = find_images_txt(out_dir.parent / f"{out_dir.name}_colmap")

    if images_txt is None:
        raise RuntimeError(
            f"Could not find COLMAP images.txt under {out_dir}.\n"
            f"Tip: pass --try_vipe_to_colmap and --vipe_repo /path/to/vipe."
        )

    tqdm.write(f"{verbose_prefix}  Parsing poses from {images_txt.name}...")

    # Cache images.txt for GT videos
    if gt_cache_path is not None:
        gt_cache_path.mkdir(parents=True, exist_ok=True)
        cached_images_txt = gt_cache_path / "images.txt"
        import shutil
        shutil.copy2(images_txt, cached_images_txt)
        tqdm.write(f"{verbose_prefix}  Cached GT trajectory to: {cached_images_txt.name}")

    poses, num_images = parse_colmap_images_txt(images_txt)
    tqdm.write(f"{verbose_prefix}  Extracted {num_images} poses")
    return poses

def bucket_for_action(a: str) -> str:
    a = a.lower()
    if a in {"w", "a", "s", "d", "forward", "backward", "left", "right"}:
        return "translation"
    if a in {"cam_left", "cam_right", "cam_up", "cam_down",
             "yaw_left", "yaw_right", "pitch_up", "pitch_down"}:
        return "rotation"
    return "other"