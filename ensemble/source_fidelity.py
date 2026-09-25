"""Fail-closed verification for vendored native sources and scientific settings."""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "native_cnn/cnn_cache.py": "266ea6a5445e04af092746ea73d0f8894f5e12c5",
    "native_cnn/cnn_model.py": "24e81ceedbfae4dec72270530cfda2ec96db9aad",
    "native_cnn/cnn_evaluation.py": "d239f7e70b172d6f14a306fab02fbe07cdf6ad1c",
    "native_mlp/cache.py": "4008833725f9239cbfa2258651c581938cc4d367",
    "native_mlp/config.py": "c8a3497fed8a0e9a75dbfc0b971790b011c92320",
    "native_mlp/data_loader.py": "c622ddd8ffe0e4ecd7a69562df2dc642a0986d24",
    "native_mlp/feature_extraction.py": "68ea0b2db91d2ec832c2c99e57ac4bd428807a73",
    "native_mlp/model.py": "956241a4179a6fec23b261d57180f359942c4b5f",
    "native_mlp/evaluation.py": "b4d69bc9bfa60bed921a050984fa5dbd1a220821",
    "native_mlp/main.py": "c118cf5dd6724e1b2bcfd1ece71380da78e13fdc",
}


def _git_source_state(repo_root, relative):
    """Return the committed blob SHA and whether the worktree path matches HEAD."""
    git_path = Path(relative).as_posix()
    committed = subprocess.run(
        ["git", "rev-parse", f"HEAD:{git_path}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", git_path],
        cwd=repo_root,
        check=False,
    ).returncode == 0
    return committed, unchanged


def _constants(path):
    tree, values = ast.parse(path.read_text()), {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try: values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError): pass
    return values


def audit(output=None):
    rows, failures = [], []
    for relative, expected in EXPECTED.items():
        repo_relative = Path("ensemble") / relative
        actual, unchanged = _git_source_state(ROOT.parent, repo_relative)
        status = "MATCH" if actual == expected and unchanged else "MISMATCH"
        rows.append(f"BLOB {relative} expected={expected} actual={actual} {status}")
        rows.append(f"WORKTREE {relative}: {'CLEAN' if unchanged else 'MODIFIED'}")
        if status != "MATCH": failures.append(relative)
    cnn_eval = _constants(ROOT / "native_cnn/cnn_evaluation.py")
    cnn_model = _constants(ROOT / "native_cnn/cnn_model.py")
    checks = {
        "CNN channel count": cnn_model.get("NUM_CHANNELS") == 35,
        "CNN context windows": cnn_eval.get("WINDOWS") == [31, 61, 91, 121],
        "CNN samples/patient": cnn_eval.get("TRAIN_SAMPLES_PER_PATIENT") == 1500,
        "CNN positive fraction": cnn_eval.get("POSITIVE_FRACTION") == .30,
        "CNN max epochs": cnn_eval.get("MAX_EPOCHS") == 12,
        "CNN patience": cnn_eval.get("PATIENCE") == 4,
        "CNN learning rate": cnn_eval.get("LEARNING_RATE") == .001,
        "CNN weight decay": cnn_eval.get("WEIGHT_DECAY") == 1e-4,
        "CNN batch size": cnn_eval.get("BATCH_SIZE") == 512,
        "CNN unweighted BCE": "nn.BCEWithLogitsLoss()" in (ROOT / "native_cnn/cnn_evaluation.py").read_text().replace("\n", "").replace("\t", ""),
        "CNN supplied QRS": 'f["QRS"]' in (ROOT / "native_cnn/cnn_cache.py").read_text(),
        "MLP mean imputer": 'strategy="mean"' in (ROOT / "native_mlp/model.py").read_text(),
        "MLP StandardScaler": "StandardScaler()" in (ROOT / "native_mlp/model.py").read_text(),
        "MLP architecture": all(token in (ROOT / "native_mlp/model.py").read_text() for token in ("nn.Linear(\n                input_features,\n                128", "nn.LayerNorm(128)", "nn.GELU()", "nn.Dropout(0.25)", "nn.LayerNorm(64)")),
    }
    for name, matched in checks.items():
        rows.append(f"SETTING {name}: {'MATCH' if matched else 'MISMATCH'}")
        if not matched: failures.append(name)
    text = "SOURCE FIDELITY AUDIT\n" + "\n".join(rows) + f"\nOVERALL: {'MATCH' if not failures else 'MISMATCH'}\n"
    output = output or ROOT / "results/source_fidelity_audit.txt"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(text)
    if failures: raise RuntimeError("Source-fidelity audit failed: " + ", ".join(failures))
    return text


if __name__ == "__main__":
    print(audit(), end="")
