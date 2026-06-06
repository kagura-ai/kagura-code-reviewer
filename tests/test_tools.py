import subprocess
from pathlib import Path

import pytest

from kagura_code_review.tools import RepoTools


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
