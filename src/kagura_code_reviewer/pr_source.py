"""Resolve a GitHub PR URL to a reviewable, sandboxed diff source.

The reviewer's quality edge is that it reads the *working tree* at the head of a
change (read_file/grep/list_files), not just the diff text. So to review a PR we
must materialize the PR head as a real on-disk working tree and keep the existing
``RepoTools`` sandbox (``tools.py``) the single, unchanged trust boundary.

Design (see issue #22):

- A *front resolver* sits before ``RepoTools`` construction. ``resolve_pr`` returns
  a ``DiffSource(repo_root, base, head, cleanup)``; the CLI then does
  ``RepoTools(src.repo_root)`` and ``git_diff(src.base, src.head)`` exactly as it
  does for a local review. ``tools.py`` and the harness are untouched.
- The PR head is fetched via the **immutable pull ref** ``refs/pull/<N>/head`` —
  it exists for open *and* closed/merged PRs, survives source-branch deletion, and
  (because GitHub copies fork head commits to the base repo) covers fork PRs
  without adding a fork remote. We do not use ``gh pr checkout`` (it mutates the
  operator's working tree/branch).
- The head is checked out into an isolated ``git worktree`` (sharing the object
  store) so the operator's current working directory is never disturbed.
- ``gh pr view`` is used only to map the URL to metadata (the PR's base branch) and
  for private-repo auth.

v1 scope: ``resolve_pr`` must run inside a clone of the PR's repository — the pull
ref is fetched from that clone's ``remote``. To stop a PR URL for a *different* repo
from silently reviewing the local origin's PR #N, the URL's ``owner/repo`` is
checked against the local remote when that remote is a recognizable GitHub URL.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4

_log = logging.getLogger(__name__)

_PR_URL_RE = re.compile(
    r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)/?$"
)
# Matches https://github.com/owner/repo(.git), git@github.com:owner/repo(.git),
# ssh://git@github.com/owner/repo. Anchored at the host so a look-alike host
# (mygithub.com) or github.com appearing as a path segment of another host does
# NOT match — otherwise the origin↔URL guard could compare the wrong owner/repo.
_GH_REMOTE_RE = re.compile(
    r"^(?:(?:https?|git|ssh)://)?(?:[^@/]+@)?github\.com[:/]"
    r"(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)


@dataclass
class DiffSource:
    """A materialized review target: a sandbox root plus the refs to diff.

    ``repo_root`` is handed to ``RepoTools`` unchanged. ``base``/``head`` are passed
    to ``RepoTools.git_diff`` (three-dot ``base...head``). ``cleanup`` tears down any
    temporary worktree/ref and is always safe to call (idempotent, never raises).
    """

    repo_root: Path
    base: str
    head: str
    cleanup: Callable[[], None]


def parse_pr_url(url: str) -> tuple[str, str, int]:
    """Parse a GitHub PR URL into (owner, repo, number). Raise ValueError otherwise."""
    m = _PR_URL_RE.match(url.strip())
    if not m:
        raise ValueError(f"not a GitHub PR URL: {url!r}")
    return m["owner"], m["repo"], int(m["number"])


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a captured, text subprocess. The single place the call shape lives so
    error handling stays consistent across pr_metadata / _git / remote probing /
    teardown."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def pr_metadata(url: str) -> dict:
    """Resolve PR metadata via ``gh pr view`` (the only network/auth boundary).

    On failure, raise ``RuntimeError`` carrying gh's stderr so the operator can see
    *why* (auth, not-found, wrong repo) instead of an opaque exit status.
    """
    proc = _run(["gh", "pr", "view", url, "--json",
                 "number,state,baseRefName,headRefOid,headRepository"])
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh pr view failed for {url}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    data = json.loads(proc.stdout)
    if not data.get("baseRefName"):
        raise RuntimeError(f"PR metadata for {url} is missing baseRefName: {data!r}")
    return data


def _git(repo_root: Path, *args: str) -> str:
    """Run git, raising RuntimeError (with stderr) on failure instead of an opaque
    CalledProcessError whose message hides the cause.

    Intentionally NOT shared with ``RepoTools._git`` (tools.py): that one is bound to
    the sandbox root and uses ``check=True`` (opaque error); this runs against an
    arbitrary clone root (before the worktree exists) and surfaces git's stderr for
    diagnosability. Keeping them separate also preserves the "tools.py unchanged"
    sandbox invariant this feature depends on."""
    proc = _run(["git", *args], repo_root)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _remote_github_repo(repo_root: Path, remote: str) -> tuple[str, str] | None:
    """Return (owner, repo) for ``remote`` if it is a GitHub URL, else None."""
    proc = _run(["git", "remote", "get-url", remote], repo_root)
    if proc.returncode != 0:
        return None
    m = _GH_REMOTE_RE.search(proc.stdout.strip())
    return (m["owner"], m["repo"]) if m else None


def _teardown(repo_root: Path, local_ref: str, worktree: Path | None,
              parent: Path | None) -> None:
    """Best-effort removal of a temp worktree, its pull ref, and temp dir.

    Never raises; logs a warning if the worktree could not be removed so stale
    state does not vanish silently.
    """
    if worktree is not None:
        proc = _run(["git", "worktree", "remove", "--force", str(worktree)], repo_root)
        if proc.returncode != 0:
            _log.warning("failed to remove --pr worktree %s: %s",
                         worktree, proc.stderr.strip())
    _run(["git", "update-ref", "-d", local_ref], repo_root)
    if parent is not None:
        shutil.rmtree(parent, ignore_errors=True)


def resolve_pr(
    url: str,
    *,
    repo_root: Path | str = ".",
    remote: str = "origin",
    keep: bool = False,
) -> DiffSource:
    """Materialize a GitHub PR head into an isolated worktree for review.

    ``repo_root`` must be a clone of the PR's repository (v1 scope). Returns a
    :class:`DiffSource` whose ``repo_root`` is the worktree; call ``cleanup()`` when
    done (the CLI does so in a ``finally``).
    """
    url_owner, url_repo, number = parse_pr_url(url)
    repo_root = Path(repo_root).resolve()

    # Guard against reviewing the wrong repo's PR #N: if the local remote is a
    # recognizable GitHub repo, it must match the URL. (A non-GitHub remote — local
    # path, mirror, SSH alias — can't be compared, so we proceed.)
    origin_repo = _remote_github_repo(repo_root, remote)
    if origin_repo is not None and (
        (origin_repo[0].lower(), origin_repo[1].lower())
        != (url_owner.lower(), url_repo.lower())
    ):
        raise ValueError(
            f"--pr URL is for {url_owner}/{url_repo} but the local '{remote}' remote "
            f"is {origin_repo[0]}/{origin_repo[1]} — run inside a clone of the PR's repo"
        )

    base_ref = pr_metadata(url)["baseRefName"]

    # A unique ref name so re-running the same PR never collides with a leftover.
    token = uuid4().hex[:8]
    local_ref = f"refs/kagura/pr-{number}-{token}"

    parent: Path | None = None
    worktree: Path | None = None
    try:
        # One fetch for both the immutable pull head and the base branch. The base
        # is qualified as refs/heads/<base> (not a bare arg) so a ref name beginning
        # with '-' can't be parsed as a git option, and is mapped to the
        # remote-tracking ref the three-dot diff needs (origin/<base>).
        _git(repo_root, "fetch", remote,
             f"refs/pull/{number}/head:{local_ref}",
             f"+refs/heads/{base_ref}:refs/remotes/{remote}/{base_ref}")
        # git worktree add refuses an existing path, so create a parent temp dir and
        # add the worktree as a fresh child of it.
        parent = Path(tempfile.mkdtemp(prefix=f"kcr-pr-{number}-"))
        worktree = parent / f"pr-{number}"
        _git(repo_root, "worktree", "add", "--detach", str(worktree), local_ref)
    except Exception:
        _teardown(repo_root, local_ref, worktree, parent)
        raise

    done = {"v": False}

    def cleanup() -> None:
        if keep or done["v"]:
            return
        done["v"] = True
        _teardown(repo_root, local_ref, worktree, parent)

    return DiffSource(
        repo_root=worktree,
        base=f"{remote}/{base_ref}",
        head=local_ref,
        cleanup=cleanup,
    )
