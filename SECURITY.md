# Security Policy

## Supported versions

This project is pre-1.0. Security fixes are applied to the latest released
version on `main`.

| Version | Supported |
| ------- | --------- |
| latest `main` / newest release | ✅ |
| older releases | ❌ |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Use GitHub's **private vulnerability reporting**: the repository's *Security* tab
→ *Report a vulnerability*
(<https://github.com/kagura-ai/kagura-code-reviewer/security/advisories/new>).

Please include:

- a description of the issue and its impact,
- steps to reproduce (a minimal diff or command is ideal), and
- any suggested remediation.

We aim to acknowledge reports within a few days and to coordinate a fix and
disclosure timeline with you through the advisory.

## Threat model notes

`kagura-code-reviewer` runs an LLM over a git diff using sandboxed repo tools
(`read_file` / `grep` / `git_diff` / `list_files`). Two areas deserve special
attention when reporting:

1. **Prompt injection via diff or injected memory context.** The diff and any
   `--context-file` memory are treated as **untrusted data**, fenced in
   `BEGIN/END UNTRUSTED` markers, and the system prompt forbids obeying
   instructions inside them. Memory has **no finding-suppression authority** —
   the verdict is computed only from `submit_findings` output and the
   adversarial verify pass. See the "Memory security contract" section of the
   README.
2. **Sandbox escape via repo tools.** `read_file` enforces a secret denylist
   (`.env*`, private keys, `.git`, etc.) and path containment. Reports of ways
   to read outside the repo or exfiltrate secrets are in scope.
