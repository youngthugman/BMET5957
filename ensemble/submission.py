"""Create and strictly validate the annotation-only MATLAB submission."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat


def load_template(path):
    raw = loadmat(path, variable_names=("Class",), squeeze_me=False)
    if "Class" not in raw: raise ValueError("Annotation template has no Class field")
    value = raw["Class"]
    if value.dtype != object: raise ValueError("Class must be a MATLAB cell array")
    return value


def create_submission(template_path, probabilities_by_patient, threshold, output_path):
    template = load_template(template_path)
    cells = np.empty(template.shape, dtype=object)
    flat_template = template.ravel(order="F")
    if len(flat_template) != 100 or len(probabilities_by_patient) != 100:
        raise ValueError("Submission requires exactly 100 patients")
    for index, (original, probability) in enumerate(zip(flat_template, probabilities_by_patient)):
        original = np.asarray(original)
        probability = np.asarray(probability).ravel()
        if probability.size != original.size: raise ValueError(f"Patient {index + 1} length mismatch")
        labels = np.where(probability >= threshold, "A", "N")
        # Reshape in MATLAB column-major order: no transpose and identical orientation.
        cell_index = np.unravel_index(index, template.shape, order="F")
        cells[cell_index] = labels.reshape(original.shape, order="F")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(output_path, {"Class": cells}, do_compression=True)
    return validate_submission(template_path, output_path)


def validate_submission(template_path, output_path, report_path=None):
    template, output = load_template(template_path), loadmat(output_path)
    checks = {"Class exists": "Class" in output,
              "no unexpected fields": set(key for key in output if not key.startswith("__")) == {"Class"}}
    result = output.get("Class")
    checks["exactly 100 patients"] = result is not None and result.size == template.size == 100
    total_expected = total_actual = 0
    if result is not None and result.size == template.size:
        for expected, actual in zip(template.ravel(order="F"), result.ravel(order="F")):
            expected, actual = np.asarray(expected), np.asarray(actual)
            checks.setdefault("shapes and orientation match", True)
            checks["shapes and orientation match"] &= expected.shape == actual.shape
            characters = np.asarray(actual).astype(str).ravel(order="F")
            checks.setdefault("only A/N and no ?", True)
            checks["only A/N and no ?"] &= bool(np.all(np.isin(characters, ("A", "N"))))
            total_expected += expected.size; total_actual += actual.size
    checks["prediction total matches"] = total_actual == total_expected
    lines = [f"{name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items()]
    lines.append(f"Overall: {'PASS' if all(checks.values()) else 'FAIL'}")
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not all(checks.values()): raise ValueError("Submission validation failed: " + ", ".join(lines))
    return checks
