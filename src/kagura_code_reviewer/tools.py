from __future__ import annotations

import re
import subprocess
from pathlib import Path


class RepoTools:
    """In-repo file/git/grep tools, sandboxed to the repository root."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()

    def _resolve(self, path: str) -> Path:
        target = (self.repo_root / path).resolve()
        if target != self.repo_root and self.repo_root not in target.parents:
            raise ValueError(f"path escapes repository root: {path}")
        return target

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def read_file(self, path: str, max_bytes: int = 20000) -> str:
        target = self._resolve(path)
        if not target.is_file():
            return f"error: not a file: {path}"
        data = target.read_text(errors="replace")
        if len(data) > max_bytes:
            return data[:max_bytes] + "\n...[truncated]"
        return data

    def grep(self, pattern: str, max_results: int = 50) -> str:
        rx = re.compile(pattern)
        hits: list[str] = []
        for rel in self.list_files():
            try:
                target = self._resolve(rel)
            except ValueError:
                continue
            try:
                for i, line in enumerate(target.read_text(errors="replace").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{rel}:{i}: {line.strip()}")
                        if len(hits) >= max_results:
                            return "\n".join(hits) + "\n...[more matches hidden]"
            except OSError:
                continue
        return "\n".join(hits) if hits else "no matches"

    def list_files(self, subdir: str = ".") -> list[str]:
        self._resolve(subdir)
        # `--` stops git from interpreting a subdir that starts with `-` as a flag.
        out = self._git("ls-files", "--", subdir)
        return [line for line in out.splitlines() if line]

    def git_diff(self, base: str, head: str = "HEAD", paths: list[str] | None = None) -> str:
        args = ["diff", f"{base}...{head}"]
        if paths:
            args += ["--", *paths]
        return self._git(*args)

    def changed_files(self, base: str, head: str = "HEAD") -> list[str]:
        out = self._git("diff", "--name-only", f"{base}...{head}")
        return [line for line in out.splitlines() if line]
