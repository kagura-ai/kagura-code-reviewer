import subprocess
from pathlib import Path

import pytest

from kagura_code_reviewer.tools import RepoTools


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "t@t.t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("print('hello')\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-m", "init")
    (tmp_path / "a.py").write_text("print('hello')\nprint('world')\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-m", "second")
    return tmp_path


def test_read_file_returns_content(repo: Path):
    tools = RepoTools(repo)
    assert "world" in tools.read_file("a.py")


def test_read_file_rejects_escape(repo: Path):
    tools = RepoTools(repo)
    with pytest.raises(ValueError):
        tools.read_file("../../etc/passwd")


def test_grep_finds_match(repo: Path):
    tools = RepoTools(repo)
    out = tools.grep("world")
    assert "a.py" in out


def test_list_files_lists_tracked(repo: Path):
    tools = RepoTools(repo)
    assert "a.py" in tools.list_files()


def test_git_diff_shows_change(repo: Path):
    tools = RepoTools(repo)
    diff = tools.git_diff("HEAD~1", "HEAD")
    assert "world" in diff


def test_changed_files_lists_path(repo: Path):
    tools = RepoTools(repo)
    assert "a.py" in tools.changed_files("HEAD~1", "HEAD")


def test_grep_does_not_follow_symlink_outside_repo(repo: Path):
    outside = repo.parent / "outside_secret.txt"
    outside.write_text("TOPSECRET_NEEDLE\n")
    link = repo / "leak.txt"
    link.symlink_to(outside)
    subprocess.run(["git", "add", "leak.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add link"], cwd=repo, check=True, capture_output=True)
    tools = RepoTools(repo)
    out = tools.grep("TOPSECRET_NEEDLE")
    assert "TOPSECRET_NEEDLE" not in out


def test_list_files_uses_dashdash_separator(tmp_path, monkeypatch):
    from kagura_code_reviewer.tools import RepoTools
    rt = RepoTools(tmp_path)
    captured = {}
    monkeypatch.setattr(rt, "_git", lambda *a: captured.setdefault("args", a) and "" or "")
    rt.list_files("--deleted")
    assert captured["args"] == ("ls-files", "--", "--deleted")


def test_read_file_marks_truncation(tmp_path):
    from kagura_code_reviewer.tools import RepoTools
    (tmp_path / "f.txt").write_text("abcdefghij")
    out = RepoTools(tmp_path).read_file("f.txt", max_bytes=4)
    assert out.startswith("abcd") and out.endswith("...[truncated]")


def test_grep_marks_capped_results(tmp_path, monkeypatch):
    from kagura_code_reviewer.tools import RepoTools
    (tmp_path / "a.txt").write_text("m\nm\nm\nm\n")
    rt = RepoTools(tmp_path)
    monkeypatch.setattr(rt, "list_files", lambda subdir=".": ["a.txt"])
    out = rt.grep("m", max_results=2)
    assert out.strip().endswith("...[more matches hidden]")
