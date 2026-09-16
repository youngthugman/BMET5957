"""Load BMET3997/9997 major-project .mat files into plain Python structures."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.io import loadmat

FS = 100  # Hz, fixed by the project brief


def _unpack_cell(cell: np.ndarray) -> list[np.ndarray]:
    """Turn a MATLAB (1, N) cell array into a list of 1-D numpy arrays."""
    out = []
    for item in cell.flatten():
        out.append(np.asarray(item).flatten())
    return out


def load_training(path: str | Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (ecg_list, qrs_expert_list). Each is a list of 35 1-D arrays."""
    m = loadmat(str(path))
    ecg = _unpack_cell(m["ECG"])
    qrs = _unpack_cell(m["QRSexpert"])
    # Cast ECG to float for downstream DSP; keep QRS as int.
    ecg = [x.astype(np.float64) for x in ecg]
    qrs = [x.astype(np.int64) for x in qrs]
    return ecg, qrs


def load_test_ecg(path: str | Path) -> list[np.ndarray]:
    """Return the 35 test ECGs as a list of float64 1-D arrays."""
    m = loadmat(str(path))
    ecg = _unpack_cell(m["ECG"])
    return [x.astype(np.float64) for x in ecg]


def load_submission_template(path: str | Path) -> dict:
    """Load ProjectTestDataAnalysis.mat so we preserve its exact variable layout."""
    m = loadmat(str(path))
    return {k: v for k, v in m.items() if not k.startswith("__")}


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    ecg, qrs = load_training(here / "data" / "ProjectTrainData.mat")
    print(f"Loaded {len(ecg)} training recordings")
    for i in range(min(3, len(ecg))):
        dur_h = len(ecg[i]) / FS / 3600
        bpm = len(qrs[i]) / (len(ecg[i]) / FS) * 60
        print(f"  rec {i}: {len(ecg[i]):>8d} samples ({dur_h:.2f} h), {len(qrs[i]):>6d} QRS (~{bpm:.1f} bpm)")
