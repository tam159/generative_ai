---
name: pyc-decompile
description: Decompile / recover readable .py source from compiled Python .pyc bytecode files. Use when the user wants to decompile, reverse, un-compile, or recover source from .pyc / compiled-Python / bytecode files, or points the skill at one or more folders containing .pyc files. Reconstructs a .py next to each .pyc and verifies each is opcode-identical to the original bytecode; handles modern CPython (3.11–3.13) that off-the-shelf decompilers (uncompyle6/decompyle3/pycdc) can't.
argument-hint: "[folder-or-file-with-pyc ...]"
---

# pyc-decompile — recover verified .py from .pyc

Recovers readable `.py` source for compiled `.pyc` files and **proves** each
result correct by compiling it and checking its opcode stream is identical to
the original bytecode. This is reliable where traditional decompilers fail
(modern CPython 3.11–3.13), because it leans on the matching interpreter +
an exact verification gate rather than a fragile decompiler.

**Targets:** the folder(s)/file(s) the user named (in their message or as
arguments). If none were given, ask which folder(s) contain the `.pyc` files.

**Setup — do this once at the start of the run:**
- `SK` = this skill's scripts dir. It is `~/.claude/skills/pyc-decompile/scripts`
  (portable form: `${CLAUDE_SKILL_DIR}/scripts`). Set: `SK=~/.claude/skills/pyc-decompile/scripts`
- `PY` = a CPython interpreter **matching the .pyc version** (see Step 1). All
  scripts run as `$PY $SK/<script>.py …`.
- Detailed bytecode→source patterns are in **[reference.md](reference.md)** —
  read it before hand-reconstructing anything non-trivial.

## Step 1 — Find targets & a matching interpreter (REQUIRED)
1. List what needs work: `$PY $SK/find_targets.py <folder> …` (only `.pyc`
   lacking a `.py` sibling; sorted by code-object count = effort proxy).
   *Default: never overwrite an existing `.py` unless the user explicitly asks.*
2. Verification only works under the **same CPython version** that built the
   `.pyc`. Pick a candidate interpreter (a project `.venv/bin/python`, pyenv,
   uv, conda, or system `python3`) and run:
   `<candidate> $SK/preflight.py <one-target.pyc>`
   Use the interpreter it confirms as `$PY`. If none match, tell the user which
   CPython version is needed and stop — do not proceed with a mismatched one.

## Step 2 — Fast path: copy a verified original if one exists
Compiled trees are often shipped next to (or mirrored from) the real source.
A copied original keeps real comments/formatting and is strictly better than a
reconstruction. For each target, search plausible source roots (repo root,
sibling `src/`/package trees, any path the user suggests):

`$PY $SK/find_source.py <target.pyc> <search-root> [<search-root> …]`

If it prints a `COPY:` path, copy that file to the target location
(`cp <source> <target.py>`) and re-verify (Step 4). Only reconstruct (Step 3)
the files with no verifying source. (Mirrors aren't always in sync — the script
already confirms an exact match, so trust only what it prints.)

## Step 3 — Reconstruct the rest (per file)
Loop until verified:
1. `$PY $SK/introspect.py <target.pyc>` → map imports, classes, signatures, consts.
2. `$PY $SK/disasm.py <target.pyc>` → the actual logic + ExceptionTable. For big
   files redirect to a temp file and Read it in ranges.
3. Write the `.py` next to the `.pyc`, working top-down (module → classes →
   methods). Apply the patterns in [reference.md](reference.md).
4. `$PY $SK/verify.py <target.pyc> <target.py>` → fix the first reported
   mismatch, repeat until it prints `✅ LOGICAL MATCH`. That is done-done.
   Comments/blank lines need not match (not in bytecode); a `LOAD_CONST <int>`
   diff in a class body is a `__firstlineno__` → pad lines to place the `class`.

## Step 4 — Scale: batch large jobs across parallel subagents
For more than a handful of files, fan out — files are independent and the
verify gate makes delegation safe (a subagent is "done" only when verify prints
`✅ LOGICAL MATCH`). Use the code-object counts from Step 1 to bin-pack into
balanced batches (~50–60 objs/batch; give very large single files their own
agent). Spawn `general-purpose` subagents (Agent tool) in waves; instruct each
to read this skill's `reference.md` + use `$SK/{introspect,disasm,verify}.py`,
decompile its assigned files, and report per-file `✅ MATCH` / remaining diff.
Then run the final audit yourself. (Only escalate to the Workflow tool if the
user has explicitly opted into multi-agent orchestration.)

## Step 5 — Final audit
`$PY $SK/verify.py --audit <folder> …` → confirms every `.pyc` has a `.py` and
every one matches (`verified=N missing=0 mismatch=0`). Also byte-compile the
outputs as a smoke test: `$PY -m compileall -q <folder>`. Report the tally,
which files were copied originals vs reconstructed, and anything left untouched.

## Rules & defaults
- Only create a `.py` where one doesn't exist; leave pre-existing `.py` alone
  unless the user says otherwise.
- Never edit the `.pyc` files or this skill's scripts.
- A result counts as correct **only** when `verify.py` prints `✅ LOGICAL MATCH`
  — never hand-wave "looks right". If a file won't converge, leave the best
  attempt and report the remaining diff verbatim.
- `.pyc` files leak nothing executable here — the scripts only read bytecode and
  compile candidate source; they never run the target code.
