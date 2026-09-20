
"""Functions for loading and preparing the MATLAB training data."""

import h5py
import numpy as np
from scipy.io import loadmat

from config import FIELDS


def _scipy_cells(value):
    """Turn a regular MAT cell array (or scalar) into a patient list."""
    value = np.asarray(value)

    if value.dtype == object:
        return [np.asarray(x).squeeze() for x in value.ravel(order="F")]

    if value.ndim <= 1:
        return [value.squeeze()]

    return [value[:, i].squeeze() for i in range(value.shape[1])]


def _hdf5_value(handle, item):
    """Read one value from a MATLAB v7.3/HDF5 file."""
    array = (
        np.asarray(handle[item])
        if isinstance(item, h5py.Reference)
        else np.asarray(item)
    )

    # MATLAB stores arrays transposed in v7.3 files.
    # Vectors are unaffected by this operation.
    return array.T.squeeze()


def _hdf5_cells(handle, name):
    """Read a MATLAB cell array from a v7.3/HDF5 file."""
    dataset = handle[name]

    if h5py.check_dtype(ref=dataset.dtype) is not None:
        return [
            _hdf5_value(handle, ref)
            for ref in dataset[()].ravel(order="F")
        ]

    array = np.asarray(dataset).T

    if array.size == 1:
        return [array.squeeze()]

    if array.ndim == 1 or 1 in array.shape:
        return [np.asarray(x) for x in array.ravel(order="F")]

    return [array[:, i].squeeze() for i in range(array.shape[1])]


def load_training_data(path):
    """Load both ordinary and MATLAB v7.3/HDF5 MAT files."""

    # MATLAB v7.3 files are HDF5 files and need to be handled differently
    # from older MAT-file formats.
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            missing = [name for name in FIELDS if name not in handle]

            if missing:
                raise ValueError(
                    f"MAT file is missing: {', '.join(missing)}"
                )

            return {
                name: _hdf5_cells(handle, name)
                for name in FIELDS
            }

    # Older/non-v7.3 MAT files can be loaded directly with scipy.
    raw = loadmat(
        path,
        variable_names=FIELDS,
        squeeze_me=False,
    )

    missing = [name for name in FIELDS if name not in raw]

    if missing:
        raise ValueError(
            f"MAT file is missing: {', '.join(missing)}"
        )

    return {
        name: _scipy_cells(raw[name])
        for name in FIELDS
    }


def scalar_rate(values, patient_index, name):
    """Get the sampling rate belonging to a particular patient."""

    item = values[patient_index] if len(values) > 1 else values[0]
    item = np.asarray(item, dtype=float).ravel()

    if (
        item.size != 1
        or not np.isfinite(item[0])
        or item[0] <= 0
    ):
        raise ValueError(
            f"Invalid {name} for patient {patient_index + 1}"
        )

    return float(item[0])


def clean_labels(value):
    """Convert MATLAB N/A annotations into binary labels.

    N -> 0
    A -> 1
    """

    array = np.asarray(value)

    if (
        array.dtype.kind in "ui"
        and array.size
        and np.nanmax(array) <= 65535
    ):
        text = "".join(
            chr(int(x))
            for x in array.ravel(order="F")
            if int(x)
        )
    else:
        text = "".join(
            str(x)
            for x in array.ravel(order="F")
        )

    labels = np.array([
        c.upper()
        for c in text
        if c.upper() in ("N", "A")
    ])

    if labels.size == 0:
        raise ValueError("No N/A annotations found")

    return (labels == "A").astype(np.uint8)
