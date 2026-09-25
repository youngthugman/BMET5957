import subprocess

from ensemble.source_fidelity import _git_source_state


def test_git_source_state_accepts_unchanged_crlf_checkout(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=repo, check=True)
    source = repo / "native.py"
    source.write_bytes(b"first = 1\nsecond = 2\n")
    subprocess.run(["git", "add", "native.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "Add native source"], cwd=repo, check=True)
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD:native.py"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    source.unlink()
    subprocess.run(["git", "checkout", "--", "native.py"], cwd=repo, check=True)

    assert source.read_bytes() == b"first = 1\r\nsecond = 2\r\n"
    assert _git_source_state(repo, "native.py") == (expected, True)

    source.write_bytes(source.read_bytes() + b"changed = True\r\n")
    assert _git_source_state(repo, "native.py") == (expected, False)
