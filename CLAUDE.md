# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Repository

`hdwm` is a research experiment for Hypothesis-Driven World Model, detailed in [`PLAN_v1.md`](PLAN_v1.md).
Project documentation is available in two languages:
- [`README.md`](README.md) — English version.
- [`README.zh.md`](README.zh.md) — Chinese version.

## Tech Stack
* Pytorch
* more to be determined

## Working Rules

### Scope

1. Prefer minimal changes first.
2. Prefer comprehensive minimal fixes over narrowly local ones.
3. Keep options simple unless the user asks for more.

### Source of Truth

1. Do not invent details when the repo already has an answer.
2. If something is underspecified, use the existing code as the default source of truth.
3. If assumptions still remain, keep them small and say them explicitly.

### Process

1. Go ahead without asking for small, local fixes.
2. For a larger patch, present a short plan and ask for permission before proceeding.
3. Review your changes against these working rules before finishing; if you find a small issue, fix it.
4. Add visibility when an operation may take noticeable time.
5. Do not leave work unfinished unless the user explicitly asks to stop, defer, or leave a placeholder.

### Maintenance

1. Update `AGENTS.md` only when a new rule is necessary and keep the rule general.
2. The project maintains bilingual documentation (`README.md` and `README.zh.md`).
   When editing one README, always synchronize the other to keep content consistent.

### Commits

1. Write concise and informative git commit messages: use a short subject line and include bulleted detail lines for the important changes.

### Communication

1. Reply to the user in Chinese, but keep technical terms in English.
2. When reporting a summary of completed work, include what was done, how it was done, and the necessary important details.
3. Keep replies concise but easy to understand.

### Coding Conventions

1. Following Python best practices.
2. DRY (Don't Repeat Yourself).
3. The code should be easy to understand and maintain, well modularized, well engineered.
4. Prefer readable, self-explanatory code over compact implementation tricks; avoid magic numbers by using named constants or clearly derived values.
5. Local code should be understandable from its immediate context, with names and structure that make the inferred behavior match the actual behavior.
6. Do not add features not required by user.
7. You should add type hints for all functions and classes. Prefer pydantic models for data validation over a dictionary, when necessary.
8. Add concise comments when logic is ambiguous, non-obvious, or performing special operations; avoid comments that merely restate the code.
9. Add necessary value checks and raise `ValueError` when a function relies on assumptions or must verify inputs are working as expected.
10. Mark code that requires human verification or is based on uncertain assumptions with `# CHECK-REQUIRED`, and explain what must be checked.
