"""Tests for pr_source — resolving a GitHub PR URL to a reviewable DiffSource.

The `gh` metadata boundary (pr_metadata) is the only thing stubbed; all git
operations run against a real local remote so the worktree + ref + diff path is
exercised end-to-end (per the gate1/CTO note: cleanup and closed-PR/branch-deleted
ref resolution must be covered by real git, not mocks).
"""
import logging
import subprocess
from pathlib import Path

import pytest

from kagura_code_reviewer import pr_source
from kagura_code_reviewer.tools import RepoTools


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _git_out(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


# --- parse_pr_url ------------------------------------------------------------

def test_parse_pr_url_extracts_owner_repo_number():
    assert pr_source.parse_pr_url(
        "https://github.com/kagura-ai/memory-cloud/pull/1013"
    ) == ("kagura-ai", "memory-cloud", 1013)


def test_parse_pr_url_tolerates_trailing_slash():
    assert pr_source.parse_pr_url(
        "https://github.com/o/r/pull/7/"
    ) == ("o", "r", 7)


def test_parse_pr_url_rejects_issue_url():
    with pytest.raises(ValueError):
        pr_source.parse_pr_url("https://github.com/o/r/issues/22")


# --- resolve_pr (real git remote, stubbed gh metadata) -----------------------

@pytest.fixture
def remote_clone(tmp_path: Path, monkeypatch):
    """A local 'origin' bare remote carrying main + a PR head published at
    refs/pull/1/head, plus a working clone whose origin points at it.

    The PR's source branch is intentionally NOT pushed as a normal branch — only
    refs/pull/1/head exists — so this also models the closed/branch-deleted case.
    """
    bare = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(bare))

    seed = tmp_path / "seed"
    _git(tmp_path, "init", "-b", "main", str(seed))
    _git(seed, "config", "user.email", "t@t.t")
    _git(seed, "config", "user.name", "t")
    (seed / "a.py").write_text("x = 1\n")
    _git(seed, "add", "a.py")
    _git(seed, "commit", "-m", "base")
    _git(seed, "remote", "add", "origin", str(bare))
    _git(seed, "push", "origin", "main")
    # PR commit, published ONLY as the immutable pull ref (no branch)
    _git(seed, "checkout", "-b", "pr-branch")
    (seed / "a.py").write_text("x = 1\ny = 2  # PR_NEEDLE\n")
    _git(seed, "add", "a.py")
    _git(seed, "commit", "-m", "pr change")
    _git(seed, "push", "origin", "HEAD:refs/pull/1/head")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(bare), str(clone))

    # Stub the gh boundary: PR #1 targets main.
    monkeypatch.setattr(
        pr_source, "pr_metadata",
        lambda url: {"number": 1, "state": "MERGED", "baseRefName": "main"},
    )
    return clone


def test_resolve_pr_diffs_pr_head_against_base(remote_clone: Path):
    src = pr_source.resolve_pr(
        "https://github.com/o/r/pull/1", repo_root=remote_clone)
    try:
        diff = RepoTools(src.repo_root).git_diff(src.base, src.head)
        assert "PR_NEEDLE" in diff
    finally:
        src.cleanup()


def test_resolve_pr_worktree_exposes_pr_head_files(remote_clone: Path):
    src = pr_source.resolve_pr(
        "https://github.com/o/r/pull/1", repo_root=remote_clone)
    try:
        # The sandbox reads the PR head working tree, not just the diff text.
        assert "PR_NEEDLE" in RepoTools(src.repo_root).read_file("a.py")
    finally:
        src.cleanup()


def test_resolve_pr_cleanup_removes_worktree_and_ref(remote_clone: Path):
    src = pr_source.resolve_pr(
        "https://github.com/o/r/pull/1", repo_root=remote_clone)
    worktree = src.repo_root
    head_ref = src.head
    assert worktree.exists()
    src.cleanup()
    assert not worktree.exists()
    # the temp ref must be gone from the parent repo
    refs = _git_out(remote_clone, "for-each-ref", "--format=%(refname)")
    assert head_ref not in refs


# --- #1: local origin must match the PR URL's repo ---------------------------

def test_remote_github_repo_parses_https_and_ssh(tmp_path: Path):
    repo = tmp_path / "r"
    _git(tmp_path, "init", str(repo))
    _git(repo, "remote", "add", "origin", "https://github.com/Owner/Repo.git")
    assert pr_source._remote_github_repo(repo, "origin") == ("Owner", "Repo")
    _git(repo, "remote", "set-url", "origin", "git@github.com:Owner/Repo.git")
    assert pr_source._remote_github_repo(repo, "origin") == ("Owner", "Repo")


def test_remote_github_repo_none_for_nongithub_remote(tmp_path: Path):
    repo = tmp_path / "r"
    _git(tmp_path, "init", str(repo))
    _git(repo, "remote", "add", "origin", str(tmp_path / "local.git"))
    assert pr_source._remote_github_repo(repo, "origin") is None


def test_resolve_pr_rejects_repo_mismatch(tmp_path: Path, monkeypatch):
    """A PR URL whose repo differs from the local origin is refused, before any
    fetch reviews the wrong repo's PR #N."""
    clone = tmp_path / "clone"
    _git(tmp_path, "init", str(clone))
    _git(clone, "remote", "add", "origin", "https://github.com/realowner/realrepo.git")
    monkeypatch.setattr(pr_source, "pr_metadata",
                        lambda url: {"baseRefName": "main"})  # must not be needed
    with pytest.raises(ValueError, match="realowner/realrepo"):
        pr_source.resolve_pr("https://github.com/other/proj/pull/1", repo_root=clone)


# --- #2: partial-failure teardown --------------------------------------------

def test_resolve_pr_cleans_up_on_partial_failure(remote_clone: Path, monkeypatch):
    """If a later fetch/worktree step fails, no refs/kagura ref is left behind."""
    monkeypatch.setattr(pr_source, "pr_metadata",
                        lambda url: {"baseRefName": "no-such-branch"})
    with pytest.raises(Exception):
        pr_source.resolve_pr("https://github.com/o/r/pull/1", repo_root=remote_clone)
    leftover = _git_out(remote_clone, "for-each-ref", "--format=%(refname)", "refs/kagura")
    assert leftover == ""


# --- #3: actionable diagnostics ----------------------------------------------

def test_pr_metadata_surfaces_gh_stderr(monkeypatch):
    class _P:
        returncode = 1
        stdout = ""
        stderr = "could not resolve to a PullRequest with the URL"
    monkeypatch.setattr(pr_source.subprocess, "run", lambda *a, **k: _P())
    with pytest.raises(RuntimeError, match="could not resolve to a PullRequest"):
        pr_source.pr_metadata("https://github.com/o/r/pull/999")


def test_pr_metadata_rejects_missing_base_ref(monkeypatch):
    class _P:
        returncode = 0
        stdout = '{"number": 1, "state": "OPEN"}'
        stderr = ""
    monkeypatch.setattr(pr_source.subprocess, "run", lambda *a, **k: _P())
    with pytest.raises(RuntimeError, match="baseRefName"):
        pr_source.pr_metadata("https://github.com/o/r/pull/1")


# --- #5: cleanup logs (does not silently swallow) a worktree-remove failure ---

def test_cleanup_logs_on_worktree_remove_failure(remote_clone: Path, monkeypatch, caplog):
    src = pr_source.resolve_pr("https://github.com/o/r/pull/1", repo_root=remote_clone)
    real_run = subprocess.run

    def flaky(cmd, **kw):
        if "worktree" in cmd and "remove" in cmd:
            class _P:
                returncode = 1
                stdout = ""
                stderr = "worktree is locked"
            return _P()
        return real_run(cmd, **kw)

    monkeypatch.setattr(pr_source.subprocess, "run", flaky)
    with caplog.at_level(logging.WARNING):
        src.cleanup()
    assert any("worktree" in r.message.lower() for r in caplog.records)
    # real teardown so the test leaves nothing behind
    real_run(["git", "worktree", "remove", "--force", str(src.repo_root)],
             cwd=remote_clone, capture_output=True, text=True)
    real_run(["git", "update-ref", "-d", src.head],
             cwd=remote_clone, capture_output=True, text=True)


def test_resolve_pr_creates_base_remote_tracking_ref(remote_clone: Path):
    """The combined fetch must populate origin/<base> so the three-dot diff resolves."""
    src = pr_source.resolve_pr(
        "https://github.com/o/r/pull/1", repo_root=remote_clone)
    try:
        sha = _git_out(remote_clone, "rev-parse", "--verify", "refs/remotes/origin/main")
        assert sha  # non-empty object id
    finally:
        src.cleanup()


def test_resolve_pr_works_without_source_branch_on_origin(remote_clone: Path):
    """Closed/branch-deleted/fork case: only refs/pull/1/head exists on origin (no
    pr-branch), yet resolve_pr still yields the PR head for review."""
    assert "pr-branch" not in _git_out(remote_clone, "branch", "-r")
    src = pr_source.resolve_pr(
        "https://github.com/o/r/pull/1", repo_root=remote_clone)
    try:
        assert "PR_NEEDLE" in RepoTools(src.repo_root).read_file("a.py")
    finally:
        src.cleanup()


def test_resolve_pr_keep_retains_worktree(remote_clone: Path):
    src = pr_source.resolve_pr(
        "https://github.com/o/r/pull/1", repo_root=remote_clone, keep=True)
    try:
        src.cleanup()  # no-op under keep
        assert src.repo_root.exists()
    finally:
        # manual teardown so the test leaves nothing behind
        subprocess.run(["git", "worktree", "remove", "--force", str(src.repo_root)],
                       cwd=remote_clone, capture_output=True)
        subprocess.run(["git", "update-ref", "-d", src.head],
                       cwd=remote_clone, capture_output=True)
